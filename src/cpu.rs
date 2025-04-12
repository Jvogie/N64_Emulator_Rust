use crate::bus::{Bus, BusError}; // Keep BusError removed here
use crate::exception::Exception;

// CP0 Register Numbers (Partial List)
mod cp0_reg {
    pub const INDEX: u8 = 0;
    pub const RANDOM: u8 = 1;
    pub const ENTRY_LO0: u8 = 2;
    pub const ENTRY_LO1: u8 = 3;
    pub const CONTEXT: u8 = 4;
    pub const PAGE_MASK: u8 = 5;
    pub const WIRED: u8 = 6;
    // 7 Reserved
    pub const BAD_VADDR: u8 = 8;
    pub const COUNT: u8 = 9; // Read-only shadow of Timer Counter
    pub const ENTRY_HI: u8 = 10;
    pub const COMPARE: u8 = 11; // Write-only shadow for Timer Interrupt
    pub const STATUS: u8 = 12;
    pub const CAUSE: u8 = 13;
    pub const EPC: u8 = 14;
    pub const PRID: u8 = 15;
    pub const CONFIG: u8 = 16;
    // ... add others as needed ...
    pub const WATCH_LO: u8 = 18;
    pub const WATCH_HI: u8 = 19;
    pub const XCONTEXT: u8 = 20;
    // ...
    pub const TAG_LO: u8 = 28;
    pub const TAG_HI: u8 = 29;
    pub const ERROR_EPC: u8 = 30;
}

/// Represents a single N64 TLB entry.
#[derive(Debug, Clone, Copy, Default, PartialEq)] // Add PartialEq
pub struct TlbEntry {
    // Corresponds to PageMask register (shifted)
    // Storing raw register values, derive helpers later
    page_mask: u32,
    // Corresponds to EntryHi register
    entry_hi: u64,
    // Corresponds to EntryLo0 register (PFN, C, D, V, G)
    entry_lo0: u64,
     // Corresponds to EntryLo1 register (PFN, C, D, V, G)
    entry_lo1: u64,
    // global: bool, // Derived
    // asid: u8,    // Derived
    // vpn2: u64,   // Derived
    // page_size: u64, // Derived
}

// Add helper methods to TlbEntry
impl TlbEntry {
    // Helper to get Page Size (mask applied later)
    #[inline(always)]
    fn page_mask_raw(&self) -> u32 {
        self.page_mask
    }

    // Helper to get VPN2 (Virtual Page Number / 2)
    #[inline(always)]
    fn vpn2(&self) -> u64 {
        // VPN2 is bits 63:13, ignoring the lowest bit (12) of the page number
        // Mask is 0xFFFFFFFFFFFFF000, but EntryHi already contains this range plus ASID.
        // Let's mask off the ASID part (lower 8 bits).
        self.entry_hi & !0xFFu64
    }

     // Helper to get ASID
    #[inline(always)]
    fn asid(&self) -> u8 {
        (self.entry_hi & 0xFF) as u8
    }

    // Helper to check global status (checks G bits in *both* Lo0 and Lo1)
    #[inline(always)]
    fn is_global(&self) -> bool {
        // MIPS spec: A page is global if and only if *both* G bits (EntryLo0[0] and EntryLo1[0]) are set.
        (self.entry_lo0 & 1 == 1) && (self.entry_lo1 & 1 == 1)
    }

    // Helpers to get EntryLo fields for a specific page (0 or 1)
    #[inline(always)]
    fn pfn(&self, page_index: usize) -> u64 { // Physical Frame Number
        let entry_lo = if page_index == 0 { self.entry_lo0 } else { self.entry_lo1 };
        (entry_lo >> 6) & 0x0FFFFFFF // PFN is bits 35:6 (mask for 30 bits in MIPS R4300i)
        // MIPS R4300i Manual, Table 5-21: PFN is bits 35..6
    }

    #[inline(always)]
    fn cache_coherency(&self, page_index: usize) -> u8 { // C bits
        let entry_lo = if page_index == 0 { self.entry_lo0 } else { self.entry_lo1 };
        ((entry_lo >> 3) & 0x7) as u8 // C is bits 5:3
    }

    #[inline(always)]
    fn is_dirty(&self, page_index: usize) -> bool { // D bit
        let entry_lo = if page_index == 0 { self.entry_lo0 } else { self.entry_lo1 };
        (entry_lo >> 2) & 1 == 1 // D is bit 2
    }

     #[inline(always)]
    fn is_valid(&self, page_index: usize) -> bool { // V bit
        let entry_lo = if page_index == 0 { self.entry_lo0 } else { self.entry_lo1 };
        (entry_lo >> 1) & 1 == 1 // V is bit 1
    }
}

// Placeholder for Coprocessor 0 (System Control) state
#[derive(Debug)] // Add Debug trait
pub struct Cp0 {
    // TODO: Add more CP0 registers (Context, BadVAddr, EntryHi/Lo, etc.)
    /// Status Register (controls interrupts, operating modes)
    status: u32,
    /// Cause Register (indicates cause of exceptions/interrupts)
    cause: u32,
    /// Exception Program Counter (address where exception occurred)
    epc: u64,
    /// Processor Revision Identifier
    prid: u32,
    /// Configuration Register
    config: u32,
    /// Bad Virtual Address Register
    badvaddr: u64,
    // TLB / MMU registers
    index: u32,
    random: u32, // Only lower 6 bits used typically for index
    entry_lo0: u64,
    entry_lo1: u64,
    context: u64,
    page_mask: u32,
    wired: u32, // Only lower 6 bits used
    entry_hi: u64,
    xcontext: u64,
    // Timer registers
    count: u32,   // Increments at PClock/2, read-only shadow
    compare: u32, // Write sets target, clears Cause.IP[7]
    // The TLB itself (32 entries)
    tlb: [TlbEntry; 32],
}

impl Cp0 {
    fn new() -> Self {
        const WIRED_RESET: u32 = 0; // N64 doesn't wire entries by default? Check this. Usually 0-8.
        const RANDOM_RESET: u32 = 31; // Random starts at highest index

        Cp0 {
            // N64 Specific Reset Values (Power-On)
            // Status: Kernel Mode, BEV=1 (Boot Exception Vectors), CU0=1 (CP0 Enabled), CU1=0 (FPU Disabled initially)
            status: 0x34000000,
            cause: 0,
            epc: 0,
            prid: 0x0B00, // R4300i Rev 2.0
            config: 0x7006E460, // Non-coherent cache, Big Endian, SysAD PAL, Reasonable Cache sizes
            badvaddr: 0,
            // Initialize TLB regs
            index: 0,
            random: RANDOM_RESET,
            entry_lo0: 0,
            entry_lo1: 0,
            context: 0,
            page_mask: 0, // Default 4KB page mask? Needs check. MIPS default is often 0.
            wired: WIRED_RESET,
            entry_hi: 0,
            xcontext: 0,
            // Initialize Timer regs
            count: 0, // Starts at 0
            compare: 0, // No interrupt initially
            // Initialize TLB entries (all invalid by default)
            // A simple way is Default::default(), assuming TlbEntry::default() makes it invalid.
            tlb: [TlbEntry::default(); 32],
        }
    }

    /// Reads the value of a CP0 register.
    pub fn read_reg(&self, reg_num: u8) -> u64 {
        match reg_num {
            cp0_reg::INDEX => self.index as u64,
            cp0_reg::RANDOM => self.random as u64, // Reads decrement random? No, only timer does.
            cp0_reg::ENTRY_LO0 => self.entry_lo0,
            cp0_reg::ENTRY_LO1 => self.entry_lo1,
            cp0_reg::CONTEXT => self.context,
            cp0_reg::PAGE_MASK => self.page_mask as u64,
            cp0_reg::WIRED => self.wired as u64,
            cp0_reg::ENTRY_HI => self.entry_hi,
            cp0_reg::XCONTEXT => self.xcontext,
            // --- Existing --- // Ensure existing regs are still covered
            cp0_reg::STATUS => self.status as u64,
            cp0_reg::CAUSE => self.cause as u64,
            cp0_reg::EPC => self.epc,
            cp0_reg::PRID => self.prid as u64,
            cp0_reg::CONFIG => self.config as u64,
            cp0_reg::BAD_VADDR => self.badvaddr,
            // Timer Registers
            cp0_reg::COUNT => self.count as u64,   // Read-only, return 32 bits zero-extended
            cp0_reg::COMPARE => self.compare as u64, // Read-only access to compare value
            // TODO: Implement reads for other registers (Count, Compare, Watch*, Tag*)
            _ => {
                eprintln!("Warning: Read from unimplemented CP0 register {}", reg_num);
                0
            }
        }
    }

    /// Writes a value to a CP0 register.
    pub fn write_reg(&mut self, reg_num: u8, value: u64) {
        let value32 = value as u32;
        match reg_num {
             cp0_reg::INDEX => self.index = value32 & 0x8000003F, // P, Index bits
             cp0_reg::RANDOM => self.random = value32 & 0x3F, // Should be read-only? MIPS spec varies. Assume writable for now.
             cp0_reg::ENTRY_LO0 => self.entry_lo0 = value,
             cp0_reg::ENTRY_LO1 => self.entry_lo1 = value,
             cp0_reg::CONTEXT => self.context = value, // TODO: Should this mask PTEBase?
             cp0_reg::PAGE_MASK => self.page_mask = value32 & 0x01FFE000, // Mask bits
             cp0_reg::WIRED => self.wired = value32 & 0x3F, // Wired index limit
             cp0_reg::ENTRY_HI => self.entry_hi = value, // ASID, VPN2
             cp0_reg::XCONTEXT => self.xcontext = value, // TODO: Should this mask R, PTEBase?
            // --- Existing --- // Ensure existing regs are still covered
            cp0_reg::STATUS => self.status = value32, // TODO: Implement mask for writeable bits
            cp0_reg::CAUSE => self.cause = (self.cause & !0x0000_0300) | (value32 & 0x0000_0300), // Allow writing IP0, IP1
            cp0_reg::EPC => self.epc = value,
            cp0_reg::PRID | cp0_reg::CONFIG => { /* Ignore write */ }
            cp0_reg::BAD_VADDR => { /* Ignore write */ }
            // Timer Registers
            cp0_reg::COUNT => { /* Ignore write - Read Only */ }
            cp0_reg::COMPARE => {
                self.compare = value32;
                // Writing Compare clears the timer interrupt pending bit
                self.cause &= !(1 << 15); // Clear Cause.IP[7] (bit 15)
            }
            // TODO: Implement writes for other registers (Compare, Count, Watch*, Tag*)
            _ => {
                 eprintln!("Warning: Write to unimplemented CP0 register {}", reg_num);
            }
        }
    }
}

// Placeholder for Coprocessor 1 (Floating-Point Unit) state
pub struct Cp1 {
    /// Floating-Point Registers (FPR0-FPR31)
    fpr: [f64; 32], // Using f64 for 64-bit Floating Point General Registers
    /// Floating-Point Control/Status Register (FCR31)
    fcsr: u32,
    /// FPU Implementation Register (FCR0)
    fir: u32,
    // TODO: Add FPU Implementation Control Register (FCR0 - FIR)
}

impl Cp1 {
    fn new() -> Self {
        Cp1 {
            // N64 Specific Reset Values
            fpr: [0.0; 32],
            fcsr: 0x01000000, // FS=1 (Flush denorms), RM=0 (Round to Nearest)
            fir: 0x00001400, // R4300i FPU implementation/revision
        }
    }
}

/// Type of memory access for address translation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AccessType {
    Instruction,
    DataLoad,
    DataStore,
}

// Represents the state of the MIPS R4300i CPU
pub struct Cpu {
    /// General Purpose Registers (R0-R31). R0 is hardwired to 0.
    gpr: [u64; 32],
    /// Program Counter
    pc: u64,
    /// HI register for multiplication/division results
    hi: u64,
    /// LO register for multiplication/division results
    lo: u64,
    /// Coprocessor 0 state
    cp0: Cp0,
    /// Coprocessor 1 (FPU) state
    cp1: Cp1,
    /// Load-Linked bit
    ll_bit: bool,
    /// Holds the target PC for the *next* instruction (accounts for delay slots)
    next_pc: u64,
}

impl Cpu {
    /// Creates a new CPU instance initialized to the N64 reset state.
    pub fn new() -> Self {
        // N64 Reset PC is 0xFFFF_FFFF_BFC0_0000 (physical)
        // which maps to 0xFFFFFFFF_BFC0_0000 (virtual KSEG1)
        const RESET_VECTOR: u64 = 0xFFFFFFFF_BFC0_0000;

        Cpu {
            gpr: [0; 32], // R0 is conceptually always 0, enforced by read/write logic later
            pc: RESET_VECTOR,
            next_pc: RESET_VECTOR.wrapping_add(4), // PC after the first instruction executes
            hi: 0,
            lo: 0,
            cp0: Cp0::new(),
            cp1: Cp1::new(),
            ll_bit: false,
        }
    }

    /// Sets the Program Counter (PC) and next_pc for testing/initialization.
    pub fn set_pc(&mut self, addr: u64) {
        self.pc = addr;
        self.next_pc = addr.wrapping_add(4); // Assume next instruction is sequential initially
    }

    /// Executes a single CPU instruction cycle.
    pub fn step(&mut self, bus: &mut dyn Bus) {
        let current_pc = self.pc;

        // Translate PC virtual address to physical address
        let physical_pc = match self.translate_vaddr(current_pc, AccessType::Instruction) {
            Ok(pa) => pa,
            Err(exception) => {
                // Exception during PC translation (TLB miss, address error, etc.)
                // BadVAddr is often set to the faulting VA (current_pc)
                // EPC needs careful handling (might be PC of branch if in delay slot)
                self.trigger_exception(exception, current_pc, Some(current_pc));
                return;
            }
        };

        // The instruction fetched here is the one at the *physical* address
        let instruction_result = bus.read_u32(physical_pc);

        // Update PC to the target decided by the *previous* instruction.
        // This happens *before* decoding the current instruction,
        // effectively executing the instruction in the delay slot.
        self.pc = self.next_pc;
        // Default next PC is PC + 4, unless the current instruction changes it (branch/jump)
        self.next_pc = self.pc.wrapping_add(4);

        let instruction = match instruction_result {
            Ok(instr) => instr,
            Err(_err) => {
                // Bus error on instruction fetch
                // We trigger the exception *before* updating PC to next_pc
                self.trigger_exception(Exception::BusFetch, current_pc, None);
                // Exception handler sets PC, so we return here to prevent further execution in this step
                return;
            }
        };

        // Decode and Execute instruction using *virtual* PC
        self.decode(instruction, bus, current_pc);

        // --- Update Timer --- 
        // R4300i increments Count every *two* PClock cycles.
        // For simplicity, increment once per instruction cycle for now.
        // This means timer runs at half speed relative to instructions, adjust if needed.
        self.cp0.count = self.cp0.count.wrapping_add(1);

        // Check if Count matches Compare
        if self.cp0.count == self.cp0.compare {
            // Set Cause.IP[7] (Timer Interrupt Pending)
            self.cp0.cause |= 1 << 15; 
            println!("Timer interrupt pending (Count == Compare)");
        }

        // --- Check for Interrupts ---
        // Interrupts are taken only if:
        // 1. Status.IE = 1 (Interrupts Enabled - bit 0)
        // 2. Status.EXL = 0 (Not in Exception Level - bit 1)
        // 3. Status.ERL = 0 (Not in Error Level - bit 2)
        // 4. There is a pending interrupt (Cause.IP bits 15:8) enabled by the mask (Status.IM bits 15:8)
        let status = self.cp0.status;
        let cause = self.cp0.cause;
        let interrupts_globally_enabled = (status & 1) == 1; // Check Status.IE (bit 0)
        let currently_in_exception = ((status >> 1) & 1) == 1 || ((status >> 2) & 1) == 1; // Check Status.EXL (bit 1) or Status.ERL (bit 2)

        if interrupts_globally_enabled && !currently_in_exception {
            let interrupt_mask = (status >> 8) & 0xFF;       // Status.IM (bits 15:8)
            let pending_interrupts = (cause >> 8) & 0xFF;   // Cause.IP (bits 15:8)
            let enabled_pending_interrupts = pending_interrupts & interrupt_mask;

            if enabled_pending_interrupts != 0 {
                // An enabled interrupt is pending!
                // We need to use the *current* PC (which points to the instruction *after* the one that just executed)
                // because the exception occurs between instructions.
                println!(
                    "Interrupt Check: Triggering Interrupt! Pending={:#04x}, Mask={:#04x}, Result={:#04x}",
                    pending_interrupts, interrupt_mask, enabled_pending_interrupts
                );
                self.trigger_exception(Exception::Interrupt, self.pc, None);
                // Exception handler updates PC/next_PC, so execution continues from the handler in the *next* step.
            }
        }
    }

    /// Triggers a CPU exception.
    fn trigger_exception(&mut self, cause: Exception, instruction_pc: u64, bad_vaddr: Option<u64>) {
        println!(
            "!!! EXCEPTION TRIGGERED: {:?} at PC={:#018x}{}",
            cause, instruction_pc, match bad_vaddr { Some(a) => format!(", BadVAddr={:#018x}", a), None => "".to_string() }
        );

        // Check for nested exception (simplistic handling for now)
        let status = self.cp0.status;
        if (status >> 1) & 1 == 1 { // Status.EXL is set
            // This is a critical situation (exception during exception handling)
            // Real hardware might reset or enter a special double-fault state.
            eprintln!("CRITICAL: Nested exception detected! (EXL was set). Halting.");
            // Halt simulation - a real system might reset
            self.pc = instruction_pc;
            self.next_pc = instruction_pc;
            return;
        }
        // TODO: Handle Status.ERL for error exceptions if implemented

        let is_delay_slot = self.pc != instruction_pc.wrapping_add(4);

        // 1. Update Cause register
        let exccode = (cause as u8) << 2; // Shift ExcCode into bits 6:2
        self.cp0.cause = (self.cp0.cause & !0x7C) | (exccode as u32);
        if is_delay_slot {
            self.cp0.cause |= 1 << 31; // Set BD (Branch Delay) bit
        } else {
            self.cp0.cause &= !(1 << 31); // Clear BD bit
        }
        // TODO: Update CE (Coprocessor Error) bits if applicable

        // 2. Set EPC
        // If exception occurred in delay slot, EPC points to the branch/jump instruction
        self.cp0.epc = if is_delay_slot {
            instruction_pc.wrapping_sub(4)
        } else {
            instruction_pc
        };

        // 3. Set BadVAddr if applicable
        if let Some(addr) = bad_vaddr {
            match cause {
                Exception::AddressLoad | Exception::AddressStore | Exception::TLBLoad | Exception::TLBStore => {
                     self.cp0.badvaddr = addr;
                }
                _ => { /* BadVAddr not set for this cause */ }
            }
        }

        // 4. Update Status register: Set EXL=1
        // This enters kernel mode, disables interrupts, etc.
        self.cp0.status |= 1 << 1; // Set Status.EXL (bit 1)

        // 5. Clear ll_bit
        self.ll_bit = false;

        // 6. Determine exception vector base address
        let base_vector = if (self.cp0.status >> 22) & 1 == 1 { // Check Status.BEV (bit 22)
            0xFFFFFFFF_BFC00200 // Boot exception vectors (Uncached KSEG1)
        } else {
            0xFFFFFFFF_80000000 // Normal exception vectors (Cached KSEG0)
        };

        // 7. Determine vector offset
        // Sticking to general exception offset 0x180 for now.
        // TLB Refill (ExcCode 2 or 3) without EXL uses offset 0x00/0x80 based on XTLB/UTLB mode.
        // Interrupt (ExcCode 0) uses offset 0x180 when BEV=1, or 0x200 when BEV=0.
        let offset = 0x180; // General exception vector offset

        // 8. Set PC to handler address
        self.pc = base_vector + offset; // PC is set immediately for the *next* fetch
        self.next_pc = self.pc.wrapping_add(4); // next_pc points after handler entry
    }

    /// Reads a value from a GPR, handling GPR 0 correctly.
    #[inline(always)]
    fn read_gpr(&self, index: u8) -> u64 {
        if index == 0 {
            0
        } else {
            self.gpr[index as usize]
        }
    }

    /// Writes a value to a GPR, handling GPR 0 correctly.
    #[inline(always)]
    fn write_gpr(&mut self, index: u8, value: u64) {
        if index != 0 {
            self.gpr[index as usize] = value;
        }
        //println!("DEBUG write_gpr: index={}, value={:#018x}", index, value);
    }

    // --- Public Accessors for Debugging/Testing ---

    /// Returns the current Program Counter (PC).
    pub fn pc(&self) -> u64 {
        self.pc
    }

    /// Returns the next Program Counter (next_pc) value.
    pub fn next_pc(&self) -> u64 {
        self.next_pc
    }

    /// Reads a GPR value for debugging purposes.
    /// Note: This bypasses the internal read_gpr logic which enforces GPR0=0.
    /// Use primarily for asserting final test states.
    pub fn read_gpr_debug(&self, index: u8) -> u64 {
        if index < 32 {
            self.gpr[index as usize]
        } else {
            // Or panic, or return an Option/Result
            eprintln!("Warning: read_gpr_debug called with invalid index {}", index);
            0
        }
    }
    // --- End Public Accessors ---

    /// Decodes and executes a 32-bit MIPS instruction word.
    fn decode(&mut self, instruction: u32, bus: &mut dyn Bus, current_pc: u64) {
        let op = opcode(instruction);
        let rs = rs(instruction);
        let rt = rt(instruction);
        let rd = rd(instruction);
        let shamt = shamt(instruction); // Shift amount for R-type shifts
        let func_code = funct(instruction); // Function code for SPECIAL/REGIMM opcodes (Renamed from funct)
        let imm = imm(instruction);     // 16-bit immediate for I-type
        let target = target(instruction); // 26-bit target for J-type

        println!(
            "Fetched: {:#010x}, Op: {:#04x}, rs: {}, rt: {}, rd: {}, shamt: {}, funct: {:#04x}, imm: {:#06x}, target: {:#08x}",
            instruction, op, rs, rt, rd, shamt, func_code, imm, target
        );

        // Start decoding based on the primary opcode
        match op {
            // --- SPECIAL --- (Opcode 0x00)
            0x00 => {
                // Further decoding based on the function code (func_code)
                match func_code {
                    // SLL (Shift Left Logical - Immediate)
                    0x00 => {
                        if instruction != 0 { // NOP handled separately
                            // Operates on lower 32 bits of rt, zero-extends result to 64 bits
                            let value = self.read_gpr(rt) as u32;
                            let result = (value << shamt) as u64;
                            self.write_gpr(rd, result);
                            // Default PC update
                            self.next_pc = self.pc.wrapping_add(4);
                        }
                        // NOP (instruction == 0) takes default next_pc = self.pc + 4
                    },
                    // SRL (Shift Right Logical - Immediate)
                    0x02 => {
                        // Operates on lower 32 bits of rt, zero-extends result to 64 bits
                        let value = self.read_gpr(rt) as u32;
                        let result = (value >> shamt) as u64;
                        self.write_gpr(rd, result);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // SRA (Shift Right Arithmetic - Immediate)
                    0x03 => {
                        // Operates on lower 32 bits of rt, sign-extends result to 64 bits
                        let value = self.read_gpr(rt) as u32;
                        let result = ((value as i32) >> shamt) as u64;
                        self.write_gpr(rd, result);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // SLLV (Shift Left Logical - Variable)
                    0x04 => {
                        let rt_val = self.read_gpr(rt) as u32;
                        let shift_amount = (self.read_gpr(rs) & 0x1F) as u32; // Lower 5 bits of rs
                        let result = (rt_val << shift_amount) as u64;
                        self.write_gpr(rd, result);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // SRLV (Shift Right Logical - Variable)
                    0x06 => {
                        let rt_val = self.read_gpr(rt) as u32;
                        let shift_amount = (self.read_gpr(rs) & 0x1F) as u32; // Lower 5 bits of rs
                        let result = (rt_val >> shift_amount) as u64;
                        self.write_gpr(rd, result);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // SRAV (Shift Right Arithmetic - Variable)
                    0x07 => {
                        let rt_val = self.read_gpr(rt) as u32;
                        let shift_amount = (self.read_gpr(rs) & 0x1F) as u32; // Lower 5 bits of rs
                        let result = ((rt_val as i32) >> shift_amount) as u64;
                        self.write_gpr(rd, result);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // JR (Jump Register)
                    0x08 => {
                        let target_addr = self.read_gpr(rs);
                        // TODO: Check if target_addr needs alignment check (MIPS III requires lower 2 bits 0)
                        // For now, assume correct or handled by subsequent fetch exception.
                        self.next_pc = target_addr;
                    },
                    // JALR (Jump and Link Register)
                    0x09 => {
                        let target_addr = self.read_gpr(rs);
                        let return_addr = self.pc.wrapping_add(4); // Address after delay slot
                        // rd is the link register, defaults to 31 if rd=0 is used (MIPS convention)
                        self.write_gpr(rd, return_addr);
                        // TODO: Check if target_addr needs alignment check
                        self.next_pc = target_addr;
                    },
                    // MFHI (Move From HI)
                    0x10 => {
                        self.write_gpr(rd, self.hi);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // MTHI (Move To HI)
                    0x11 => {
                        self.hi = self.read_gpr(rs);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // MFLO (Move From LO)
                    0x12 => {
                        self.write_gpr(rd, self.lo);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // MTLO (Move To LO)
                    0x13 => {
                        self.lo = self.read_gpr(rs);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // MULT (Multiply)
                    0x18 => {
                        let rs_val = (self.read_gpr(rs) as i32) as i64;
                        let rt_val = (self.read_gpr(rt) as i32) as i64;
                        let result = rs_val * rt_val;
                        self.lo = (result & 0xFFFFFFFF) as u64;
                        self.hi = (result >> 32) as u64;
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // MULTU (Multiply Unsigned)
                    0x19 => {
                        let rs_val = (self.read_gpr(rs) as u32) as u64;
                        let rt_val = (self.read_gpr(rt) as u32) as u64;
                        let result = rs_val * rt_val;
                        self.lo = (result & 0xFFFFFFFF) as u64;
                        self.hi = (result >> 32) as u64;
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // DIV (Divide)
                    0x1A => {
                        let dividend = self.read_gpr(rs) as i32;
                        let divisor = self.read_gpr(rt) as i32;
                        if divisor == 0 {
                            // Division by zero behavior is undefined on MIPS
                            // TODO: Verify N64 specific behavior for division by zero
                        } else {
                            self.lo = (dividend / divisor) as i64 as u64;
                            self.hi = (dividend % divisor) as i64 as u64;
                        }
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // DIVU (Divide Unsigned)
                    0x1B => {
                        let dividend = self.read_gpr(rs) as u32;
                        let divisor = self.read_gpr(rt) as u32;
                         if divisor == 0 {
                            // Division by zero behavior is undefined on MIPS
                            // TODO: Verify N64 specific behavior for division by zero
                        } else {
                            self.lo = (dividend / divisor) as u64;
                            self.hi = (dividend % divisor) as u64;
                        }
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // DMULT (Doubleword Multiply)
                    0x1C => {
                        let rs_val = self.read_gpr(rs) as i128;
                        let rt_val = self.read_gpr(rt) as i128;
                        let result = rs_val * rt_val;
                        self.lo = (result & 0xFFFFFFFFFFFFFFFF) as u64;
                        self.hi = (result >> 64) as u64;
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // DMULTU (Doubleword Multiply Unsigned)
                    0x1D => {
                        let rs_val = self.read_gpr(rs) as u128;
                        let rt_val = self.read_gpr(rt) as u128;
                        let result = rs_val * rt_val;
                        self.lo = (result & 0xFFFFFFFFFFFFFFFF) as u64;
                        self.hi = (result >> 64) as u64;
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // DDIV (Doubleword Divide)
                    0x1E => {
                        let dividend = self.read_gpr(rs) as i64;
                        let divisor = self.read_gpr(rt) as i64;
                        if divisor == 0 {
                            // TODO: Verify N64 behavior
                        } else {
                            self.lo = (dividend / divisor) as u64;
                            self.hi = (dividend % divisor) as u64;
                        }
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // DDIVU (Doubleword Divide Unsigned)
                    0x1F => {
                        let dividend = self.read_gpr(rs);
                        let divisor = self.read_gpr(rt);
                        if divisor == 0 {
                           // TODO: Verify N64 behavior
                        } else {
                            self.lo = dividend / divisor;
                            self.hi = dividend % divisor;
                        }
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    0x20 => { 
                        let rs_val = self.read_gpr(rs) as i32;
                        let rt_val = self.read_gpr(rt) as i32;
                        match rs_val.checked_add(rt_val) {
                            Some(result) => {
                                // Result is sign-extended to 64 bits
                                self.write_gpr(rd, result as i64 as u64);
                                self.next_pc = self.pc.wrapping_add(4);
                            }
                            None => {
                                // Overflow occurred
                                self.trigger_exception(Exception::Overflow, current_pc, None);
                                // Exception handler will set next PC
                            }
                        }
                    },
                    // ADDU (Add Unsigned) - Non-trapping 64-bit
                    0x21 => {
                        let rs_val = self.read_gpr(rs);
                        let rt_val = self.read_gpr(rt);
                        self.write_gpr(rd, rs_val.wrapping_add(rt_val));
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // SUB (Subtract with Overflow) - Operates on 32 bits
                    0x22 => { 
                        let rs_val = self.read_gpr(rs) as i32;
                        let rt_val = self.read_gpr(rt) as i32;
                        match rs_val.checked_sub(rt_val) {
                            Some(result) => {
                                // Result is sign-extended to 64 bits
                                self.write_gpr(rd, result as i64 as u64);
                                self.next_pc = self.pc.wrapping_add(4);
                            }
                            None => {
                                // Overflow occurred
                                self.trigger_exception(Exception::Overflow, current_pc, None);
                            }
                        }
                    },
                    // AND (Bitwise AND)
                    0x24 => {
                        let rs_val = self.read_gpr(rs);
                        let rt_val = self.read_gpr(rt);
                        self.write_gpr(rd, rs_val & rt_val);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // OR (Bitwise OR)
                    0x25 => {
                        let rs_val = self.read_gpr(rs);
                        let rt_val = self.read_gpr(rt);
                        self.write_gpr(rd, rs_val | rt_val);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // XOR (Bitwise XOR)
                    0x26 => {
                        let rs_val = self.read_gpr(rs);
                        let rt_val = self.read_gpr(rt);
                        self.write_gpr(rd, rs_val ^ rt_val);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // NOR (Bitwise NOR)
                    0x27 => {
                        let rs_val = self.read_gpr(rs);
                        let rt_val = self.read_gpr(rt);
                        self.write_gpr(rd, !(rs_val | rt_val));
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // SLT (Set on Less Than - Signed)
                    0x2A => {
                        let rs_val = self.read_gpr(rs) as i64;
                        let rt_val = self.read_gpr(rt) as i64;
                        let result = if rs_val < rt_val { 1 } else { 0 };
                        self.write_gpr(rd, result);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // SLTU (Set on Less Than Unsigned)
                    0x2B => {
                        let rs_val = self.read_gpr(rs);
                        let rt_val = self.read_gpr(rt);
                        let result = if rs_val < rt_val { 1 } else { 0 };
                        self.write_gpr(rd, result);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // DADD (Doubleword Add - Trapping) - Placeholder
                    0x30 => { /* DADD */
                        // Requires checking for signed overflow.
                        eprintln!("Warning: DADD (trapping) not fully implemented, behaving like DADDU.");
                        let rs_val = self.read_gpr(rs);
                        let rt_val = self.read_gpr(rt);
                        // TODO: Implement overflow check and exception trigger
                        self.write_gpr(rd, rs_val.wrapping_add(rt_val));
                        self.next_pc = self.pc.wrapping_add(4);
                     },
                    // DADDU (Doubleword Add Unsigned - Non-trapping)
                    0x31 => {
                        let rs_val = self.read_gpr(rs);
                        let rt_val = self.read_gpr(rt);
                        self.write_gpr(rd, rs_val.wrapping_add(rt_val));
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // DSUB (Doubleword Subtract - Trapping) - Placeholder
                    0x32 => {
                        // Requires checking for signed overflow.
                        eprintln!("Warning: DSUB (trapping) not fully implemented, behaving like DSUBU.");
                        let rs_val = self.read_gpr(rs);
                        let rt_val = self.read_gpr(rt);
                        // TODO: Implement overflow check and exception trigger
                        self.write_gpr(rd, rs_val.wrapping_sub(rt_val));
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // DSUBU (Doubleword Subtract Unsigned - Non-trapping)
                    0x33 => {
                        let rs_val = self.read_gpr(rs);
                        let rt_val = self.read_gpr(rt);
                        self.write_gpr(rd, rs_val.wrapping_sub(rt_val));
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // DSLL (Doubleword Shift Left Logical - Immediate)
                    0x38 => {
                        let rt_val = self.read_gpr(rt);
                        let result = rt_val << shamt;
                        self.write_gpr(rd, result);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // DSRL (Doubleword Shift Right Logical - Immediate)
                    0x3A => {
                        let rt_val = self.read_gpr(rt);
                        let result = rt_val >> shamt;
                        self.write_gpr(rd, result);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // DSRA (Doubleword Shift Right Arithmetic - Immediate)
                    0x3B => {
                        let rt_val = self.read_gpr(rt);
                        let result = ((rt_val as i64) >> shamt) as u64;
                        self.write_gpr(rd, result);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // DSLLV (Doubleword Shift Left Logical - Variable)
                    0x3C => {
                        let rt_val = self.read_gpr(rt);
                        let shift_amount = self.read_gpr(rs) & 0x3F; // Lower 6 bits of rs
                        let result = rt_val << shift_amount;
                        self.write_gpr(rd, result);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // DSRLV (Doubleword Shift Right Logical - Variable)
                    0x3E => {
                        let rt_val = self.read_gpr(rt);
                        let shift_amount = self.read_gpr(rs) & 0x3F; // Lower 6 bits of rs
                        let result = rt_val >> shift_amount;
                        self.write_gpr(rd, result);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // DSRAV (Doubleword Shift Right Arithmetic - Variable)
                    0x3F => {
                        let rt_val = self.read_gpr(rt);
                        let shift_amount = self.read_gpr(rs) & 0x3F; // Lower 6 bits of rs
                        let result = ((rt_val as i64) >> shift_amount) as u64;
                        self.write_gpr(rd, result);
                        self.next_pc = self.pc.wrapping_add(4);
                    },
                    // SYSCALL
                    0x0C => { 
                        self.trigger_exception(Exception::Syscall, current_pc, None);
                    },
                    // BREAK
                    0x0D => { 
                         self.trigger_exception(Exception::Breakpoint, current_pc, None);
                    },
                    // ... other SPECIAL opcodes ...
                    _ => { 
                        // Unimplemented SPECIAL function code
                        self.trigger_exception(Exception::ReservedInstruction, current_pc, None);
                    }
                }
            }
            // --- REGIMM --- (Opcode 0x01)
            0x01 => {
                // Further decoding based on the 'rt' field (used as a sub-opcode)
                match rt {
                    // BLTZ (Branch on Less Than Zero)
                    0x00 => { 
                        let rs_val = self.read_gpr(rs) as i64;
                        if rs_val < 0 {
                            let offset = ((imm as i16) as i64) << 2;
                            self.next_pc = self.pc.wrapping_add(offset as u64);
                        }
                        // else: PC already updated to pc + 4 (default)
                    },
                    // BGEZ (Branch on Greater Than or Equal Zero)
                    0x01 => { 
                        let rs_val = self.read_gpr(rs) as i64;
                         if rs_val >= 0 {
                            let offset = ((imm as i16) as i64) << 2;
                            self.next_pc = self.pc.wrapping_add(offset as u64);
                        }
                    },
                    // BLTZAL (Branch on Less Than Zero And Link)
                    0x10 => { 
                        let rs_val = self.read_gpr(rs) as i64;
                        let return_addr = self.pc.wrapping_add(4); // PC after delay slot
                        self.write_gpr(31, return_addr); // Link RA
                        if rs_val < 0 {
                            let offset = ((imm as i16) as i64) << 2;
                            self.next_pc = self.pc.wrapping_add(offset as u64);
                        }
                    },
                    // BGEZAL (Branch on Greater Than or Equal Zero And Link)
                    0x11 => { 
                        let rs_val = self.read_gpr(rs) as i64;
                        let return_addr = self.pc.wrapping_add(4); // PC after delay slot
                        self.write_gpr(31, return_addr); // Link RA
                         if rs_val >= 0 {
                            let offset = ((imm as i16) as i64) << 2;
                            self.next_pc = self.pc.wrapping_add(offset as u64);
                        }
                    },
                    // ... other REGIMM opcodes ...
                    _ => { 
                       // Unimplemented REGIMM rt code
                       self.trigger_exception(Exception::ReservedInstruction, current_pc, None);
                    }
                }
                // Note: PC update for branches handled inside specific instructions
            }
            // --- J-Type --- (Jump)
            0x02 => { // J (Jump)
                // Target address calculation: (PC+4 top 4 bits) | (target << 2)
                // Since self.pc is already PC+4, we use its top 4 bits
                let target_addr = (self.pc & 0xFFFFFFFF_F0000000) | ((target as u64) << 2);
                self.next_pc = target_addr;
            },
            0x03 => { // JAL (Jump And Link)
                // Target address calculation: (PC+4 top 4 bits) | (target << 2)
                let target_addr = (self.pc & 0xFFFFFFFF_F0000000) | ((target as u64) << 2);
                let return_addr = self.pc.wrapping_add(4); // PC after delay slot
                self.write_gpr(31, return_addr); // r31 (RA) gets return address
                self.next_pc = target_addr;
            },
            // --- I-Type --- (Branch, Immediate, Load/Store)
            0x04 => { 
                let rs_val = self.read_gpr(rs);
                let rt_val = self.read_gpr(rt);
                if rs_val == rt_val {
                    let offset = ((imm as i16) as i64) << 2; // Sign-extend and scale offset
                    self.next_pc = self.pc.wrapping_add(offset as u64);
                }
                 // else: PC already updated to pc + 4 (default)
            },
            0x05 => { 
                let rs_val = self.read_gpr(rs);
                let rt_val = self.read_gpr(rt);
                if rs_val != rt_val {
                    let offset = ((imm as i16) as i64) << 2;
                    self.next_pc = self.pc.wrapping_add(offset as u64);
                }
            },
            0x06 => { 
                let rs_val = self.read_gpr(rs) as i64;
                if rs_val <= 0 {
                    let offset = ((imm as i16) as i64) << 2;
                    self.next_pc = self.pc.wrapping_add(offset as u64);
                }
            },
            0x07 => { 
                let rs_val = self.read_gpr(rs) as i64;
                if rs_val > 0 {
                    let offset = ((imm as i16) as i64) << 2;
                    self.next_pc = self.pc.wrapping_add(offset as u64);
                }
            },
            0x08 => { 
                let rs_val = self.read_gpr(rs) as i32;
                let imm_val = imm as i16 as i32;
                match rs_val.checked_add(imm_val) {
                    Some(result) => {
                        // Result needs to be sign-extended back to 64 bits
                        self.write_gpr(rt, result as i64 as u64);
                        self.next_pc = self.pc.wrapping_add(4);
                    }
                    None => {
                        // Overflow occurred
                        self.trigger_exception(Exception::Overflow, current_pc, None);
                    }
                }
            },
            // ADDIU (Add Immediate Unsigned) - Non-trapping 64-bit
            0x09 => {
                let rs_val = self.read_gpr(rs);
                let imm_val = (imm as i16) as i64 as u64; // Sign-extend immediate
                self.write_gpr(rt, rs_val.wrapping_add(imm_val));
                self.next_pc = self.pc.wrapping_add(4);
            },
            // SLTI (Set on Less Than Immediate - Signed)
            0x0A => { /* SLTI */
                let rs_val = self.read_gpr(rs) as i64;
                let imm_val = (imm as i16) as i64; // Sign-extend immediate
                let result = if rs_val < imm_val { 1 } else { 0 };
                self.write_gpr(rt, result);
                self.next_pc = self.pc.wrapping_add(4);
             },
            // SLTIU (Set on Less Than Immediate Unsigned)
            0x0B => { /* SLTIU */
                let rs_val = self.read_gpr(rs);
                // Comparison is unsigned, but immediate is sign-extended first per MIPS spec
                let imm_val = (imm as i16) as i64 as u64;
                let result = if rs_val < imm_val { 1 } else { 0 };
                self.write_gpr(rt, result);
                self.next_pc = self.pc.wrapping_add(4);
            },
            // ANDI (AND Immediate)
            0x0C => { /* ANDI */
                let rs_val = self.read_gpr(rs);
                let imm_val = imm as u64; // Zero-extend immediate
                self.write_gpr(rt, rs_val & imm_val);
                self.next_pc = self.pc.wrapping_add(4);
            },
            // ORI (OR Immediate)
            0x0D => { /* ORI */
                let rs_val = self.read_gpr(rs);
                let imm_val = imm as u64; // Zero-extend immediate
                self.write_gpr(rt, rs_val | imm_val);
                self.next_pc = self.pc.wrapping_add(4);
            },
            // XORI (XOR Immediate)
            0x0E => { /* XORI */
                let rs_val = self.read_gpr(rs);
                let imm_val = imm as u64; // Zero-extend immediate
                self.write_gpr(rt, rs_val ^ imm_val);
                self.next_pc = self.pc.wrapping_add(4);
            },
            // LUI (Load Upper Immediate)
            0x0F => {
                // rt = (imm << 16) sign-extended to 64 bits
                let value = ((imm as i16) as i64 as u64) << 16;
                self.write_gpr(rt, value);
                self.next_pc = self.pc.wrapping_add(4);
            },
            // --- Coprocessor Opcodes --- (0x10-0x13)
            0x10 => { // COP0
                // Check if Status.CU[0] is enabled (usually always true for N64)
                if (self.cp0.status >> 28) & 1 == 0 {
                    self.trigger_exception(Exception::CoprocessorUnusable, current_pc, None);
                    return; // Stop processing after exception
                }

                // Dispatch based on CO bit (rs bit 4 -> instruction bit 25) and then specific fields
                if (instruction >> 25) & 1 != 0 { // Check CO bit (bit 25)
                    // CO=1: Operations dispatched by function code (bits 5:0)
                    match func_code {
                        // TLBR (Read Indexed TLB Entry)
                        0x01 => {
                            let idx = (self.cp0.index & 0x1F) as usize; // Use lower 5 bits for index
                            if idx < 32 { // Ensure index is valid
                                let entry = &self.cp0.tlb[idx];
                                self.cp0.entry_hi = entry.entry_hi;
                                self.cp0.entry_lo0 = entry.entry_lo0;
                                self.cp0.entry_lo1 = entry.entry_lo1;
                                self.cp0.page_mask = entry.page_mask;
                                println!("TLBR: Read TLB index {}", idx);
                            } else {
                                // MIPS spec says behavior is undefined for index >= TLB size.
                                // We'll just log it for now. Some emulators might ignore.
                                eprintln!("TLBR: Index {} out of bounds (>= 32)", idx);
                            }
                             self.next_pc = self.pc.wrapping_add(4);
                        }
                        // TLBWI (Write Indexed TLB Entry)
                        0x02 => {
                             let idx = (self.cp0.index & 0x1F) as usize;
                             if idx < 32 {
                                self.cp0.tlb[idx] = TlbEntry {
                                    page_mask: self.cp0.page_mask,
                                    entry_hi: self.cp0.entry_hi,
                                    entry_lo0: self.cp0.entry_lo0,
                                    entry_lo1: self.cp0.entry_lo1,
                                };
                                println!("TLBWI: Wrote TLB index {}", idx);
                             } else {
                                 // MIPS spec says behavior is undefined for index >= TLB size.
                                 eprintln!("TLBWI: Index {} out of bounds (>= 32)", idx);
                             }
                             self.next_pc = self.pc.wrapping_add(4);
                        }
                        // TLBWR (Write Random TLB Entry)
                        0x06 => {
                            // Random register should point to an entry >= Wired index
                            let wired_idx = (self.cp0.wired & 0x1F) as usize; // Wired boundary (exclusive upper limit for random writes)
                            let random_idx = (self.cp0.random & 0x1F) as usize; // Candidate index for write

                            // The Random register counts *down* from 31 to Wired.
                            // TLBWR uses the *current* Random value.
                            // Ensure the index is within the valid random range (wired..31).
                            let effective_idx = if random_idx < wired_idx {
                                // This shouldn't happen if Random is managed correctly (decrements stop at Wired)
                                // If it does, MIPS behavior is undefined. We'll write to the Wired boundary.
                                eprintln!("TLBWR: Warning - Random index ({}) < Wired index ({}). Forcing write to wired boundary.", random_idx, wired_idx);
                                wired_idx
                            } else {
                                random_idx
                            };

                            if effective_idx < 32 {
                                self.cp0.tlb[effective_idx] = TlbEntry {
                                    page_mask: self.cp0.page_mask,
                                    entry_hi: self.cp0.entry_hi,
                                    entry_lo0: self.cp0.entry_lo0,
                                    entry_lo1: self.cp0.entry_lo1,
                                };
                                println!("TLBWR: Wrote TLB index {} (Random)", effective_idx);
                            } else {
                                // Should not happen with 32 entries and 5-bit index
                                eprintln!("TLBWR: Invalid effective random index {}?", effective_idx);
                            }
                             self.next_pc = self.pc.wrapping_add(4);
                        }
                        // TLBP (Probe TLB for Matching Entry)
                        0x08 => {
                            // Probes the TLB using EntryHi (VPN2, ASID)
                            // Sets Index register: Index field = matching entry index, P bit = 0 (match) or 1 (no match)
                            let probe_vpn2 = self.cp0.entry_hi & !0x1FFFu64; // Mask VA[63:13] (ignore ASID and low bits)
                            let probe_asid = (self.cp0.entry_hi & 0xFF) as u8;
                            let mut found = false;

                            for i in 0..32 {
                                let entry = &self.cp0.tlb[i];
                                let entry_vpn2 = entry.entry_hi & !0x1FFFu64; // Mask VA[63:13]
                                let entry_asid = (entry.entry_hi & 0xFF) as u8;
                                let is_entry_global = (entry.entry_lo0 & 1 == 1) && (entry.entry_lo1 & 1 == 1); // Check G bit in *both* Lo0 and Lo1

                                // Calculate the mask to apply based on the entry's PageMask
                                let page_mask_bits = entry.page_mask_raw(); // PageMask[24:13]
                                // Mask for VA[63:13+N] where N = # of ones in low bits of PageMask + 1
                                // Simplifies to: mask everything below bit 13 + log2(page size / 4k)
                                let vpn_comparison_mask = !((page_mask_bits | 0x1FFF) as u64);

                                // Compare masked VPN
                                if (probe_vpn2 & vpn_comparison_mask) == (entry_vpn2 & vpn_comparison_mask) {
                                    // VPN matches for this page size. Now check ASID/Global.
                                    // Match occurs if:
                                    // 1. The entry is global OR
                                    // 2. The ASIDs match
                                    if is_entry_global || (probe_asid == entry_asid) {
                                        // Match found! Set Index register (Index field = i, P bit = 0)
                                        self.cp0.index = (i as u32) & 0x3F; // P bit is automatically 0
                                        println!("TLBP: Match found at index {}", i);
                                        found = true;
                                        break; // Stop searching
                                    }
                                }
                            }

                            if !found {
                                // No match found. Set P bit (Probe Failure) in Index register.
                                self.cp0.index |= 0x80000000;
                                println!("TLBP: No match found");
                            }
                            self.next_pc = self.pc.wrapping_add(4);
                        }
                        // ERET (Return From Exception)
                        0x18 => {
                            // ERET logic depends on Status.ERL and Status.EXL
                            let status = self.cp0.status;
                            if (status >> 2) & 1 == 1 { // Check Status.ERL (Error Level)
                                // Return from error: PC = ErrorEPC, Status.ERL = 0
                                println!("ERET: Returning from Error Level (ERL)");
                                self.next_pc = self.cp0.read_reg(cp0_reg::ERROR_EPC); // Read ErrorEPC (reg 30)
                                self.cp0.status &= !(1 << 2); // Clear ERL
                            } else {
                                // Return from exception: PC = EPC, Status.EXL = 0
                                println!("ERET: Returning from Exception Level (EXL)");
                                self.next_pc = self.cp0.epc;
                                self.cp0.status &= !(1 << 1); // Clear EXL
                            }
                            // ERET clears the LL bit regardless of which level we return from
                            self.ll_bit = false;
                        }
                        _ => {
                            eprintln!("Unimplemented/Invalid COP0 function code (CO=1): {:#04x}", func_code);
                            self.trigger_exception(Exception::ReservedInstruction, current_pc, None);
                        }
                    }
                } else {
                    // CO=0: Operations dispatched by rs field (bits 25:21)
                    match rs {
                         // MFC0 (Move From Coprocessor 0)
                         0x00 => {
                            // TODO: Handle potential hazards/delays if needed
                            let value = self.cp0.read_reg(rd); // Read full 64-bit CP0 register
                            // MIPS III: Result is sign-extended 32-bit value if target GPR is 32-bit context?
                            // R4300i is 64-bit, GPRs are 64-bit. Let's assume direct 64-bit move.
                            // Check if specific registers need sign-extension (e.g., Status, Cause).
                            // For now, direct move. Revisit if needed.
                            self.write_gpr(rt, value);
                            self.next_pc = self.pc.wrapping_add(4);
                        }
                         // DMFC0 (Doubleword Move From Coprocessor 0)
                         0x01 => {
                             let value = self.cp0.read_reg(rd); // Read full 64-bit register
                             self.write_gpr(rt, value);
                             self.next_pc = self.pc.wrapping_add(4);
                         }
                        // MTC0 (Move To Coprocessor 0)
                        0x04 => {
                            // TODO: Handle potential hazards/delays if needed
                            let value = self.read_gpr(rt);
                            self.cp0.write_reg(rd, value);
                            self.next_pc = self.pc.wrapping_add(4);
                        }
                         // DMTC0 (Doubleword Move To Coprocessor 0)
                         0x05 => {
                             let value = self.read_gpr(rt);
                             self.cp0.write_reg(rd, value);
                             self.next_pc = self.pc.wrapping_add(4);
                         }
                         _ => {
                            eprintln!("Unimplemented/Invalid COP0 rs opcode (CO=0): {:#04x}", rs);
                            self.trigger_exception(Exception::ReservedInstruction, current_pc, None);
                        }
                    }
                }
            }, // End COP0
            0x11 => { /* COP1 */ 
                // TODO: Further decoding needed for COP1
                self.next_pc = self.pc.wrapping_add(4);
            }, 
            0x12 => { /* COP2 */ 
                // Typically unused on N64
                 // TODO: Trigger Reserved Instruction Exception
                println!("Unimplemented Coprocessor 2 instruction");
                self.next_pc = self.pc.wrapping_add(4);
            }, 
            // 0x13 => COP3 (Reserved)
            // --- Likely Branches ---
            0x14 => { /* BEQL (Branch on Equal Likely) */ 
                self.next_pc = self.pc.wrapping_add(4);
            },
            0x15 => { /* BNEL (Branch on Not Equal Likely) */ 
                self.next_pc = self.pc.wrapping_add(4);
            },
            0x16 => { /* BLEZL (Branch on Less Than or Equal Zero Likely) */ 
                self.next_pc = self.pc.wrapping_add(4);
            },
            0x17 => { /* BGTZL (Branch on Greater Than Zero Likely) */ 
                self.next_pc = self.pc.wrapping_add(4);
            },
            // --- 64-bit Immediate Instructions ---
            0x18 => { /* DADDI (Doubleword Add Immediate) */ },
            0x19 => { /* DADDIU (Doubleword Add Immediate Unsigned) */
                //println!("DEBUG: === Entered DADDIU Handler ==="); 
                let rs_val = self.read_gpr(rs);
                let imm_val = (imm as i16) as i64 as u64; // Sign-extend immediate
                self.write_gpr(rt, rs_val.wrapping_add(imm_val));
                self.next_pc = self.pc.wrapping_add(4);
            },
            // --- Load/Store --- (Partial List)
            0x20 => { // LB
                let base = self.read_gpr(rs);
                let offset = (imm as i16) as u64; // Sign-extend offset
                let vaddr = base.wrapping_add(offset);
                match self.translate_vaddr(vaddr, AccessType::DataLoad) {
                    Ok(paddr) => {
                        // Alignment check not required for LB
                        match bus.read_u8(paddr) {
                            Ok(byte) => {
                                let value = (byte as i8) as u64; // Sign-extend byte
                                self.write_gpr(rt, value);
                                self.next_pc = self.pc.wrapping_add(4);
                            }
                            Err(_err) => { self.trigger_exception(Exception::BusLoadStore, current_pc, None); }
                        }
                    }
                    Err(exception) => { self.trigger_exception(exception, current_pc, Some(vaddr)); }
                }
            },
            0x21 => { // LH
                 let base = self.read_gpr(rs);
                 let offset = (imm as i16) as u64; // Sign-extend offset
                 let vaddr = base.wrapping_add(offset);
                 match self.translate_vaddr(vaddr, AccessType::DataLoad) {
                    Ok(paddr) => {
                         if vaddr % 2 != 0 { // Check alignment on VA
                             self.trigger_exception(Exception::AddressLoad, current_pc, Some(vaddr));
                         } else {
                             match bus.read_u16(paddr) {
                                Ok(halfword) => {
                                    let value = (halfword as i16) as u64; // Sign-extend halfword
                                    self.write_gpr(rt, value);
                                    self.next_pc = self.pc.wrapping_add(4);
                                }
                                Err(_err) => { self.trigger_exception(Exception::BusLoadStore, current_pc, None); }
                            }
                         }
                    }
                    Err(exception) => { self.trigger_exception(exception, current_pc, Some(vaddr)); }
                 }
            },
            0x23 => { // LW - Already updated as example
                 let base = self.read_gpr(rs);
                 let offset = (imm as i16) as u64; // Sign-extend offset
                 let vaddr = base.wrapping_add(offset);
                 match self.translate_vaddr(vaddr, AccessType::DataLoad) {
                     Ok(paddr) => {
                         if vaddr % 4 != 0 { 
                             self.trigger_exception(Exception::AddressLoad, current_pc, Some(vaddr));
                         } else {
                            match bus.read_u32(paddr) {
                                Ok(word) => {
                                    let value = (word as i32) as u64; // Sign-extend word
                                    self.write_gpr(rt, value);
                                    self.next_pc = self.pc.wrapping_add(4);
                                }
                                Err(_err) => { self.trigger_exception(Exception::BusLoadStore, current_pc, None); }
                            }
                        }
                     }
                     Err(exception) => { self.trigger_exception(exception, current_pc, Some(vaddr)); }
                 }
            },
            0x24 => { // LBU
                let base = self.read_gpr(rs);
                let offset = (imm as i16) as u64; // Sign-extend offset
                let vaddr = base.wrapping_add(offset);
                 match self.translate_vaddr(vaddr, AccessType::DataLoad) {
                    Ok(paddr) => {
                        match bus.read_u8(paddr) {
                            Ok(byte) => {
                                let value = byte as u64; // Zero-extend byte
                                self.write_gpr(rt, value);
                                self.next_pc = self.pc.wrapping_add(4);
                            }
                            Err(_err) => { self.trigger_exception(Exception::BusLoadStore, current_pc, None); }
                        }
                    }
                    Err(exception) => { self.trigger_exception(exception, current_pc, Some(vaddr)); }
                }
            },
            0x25 => { // LHU
                 let base = self.read_gpr(rs);
                 let offset = (imm as i16) as u64; // Sign-extend offset
                 let vaddr = base.wrapping_add(offset);
                 match self.translate_vaddr(vaddr, AccessType::DataLoad) {
                    Ok(paddr) => {
                        if vaddr % 2 != 0 {
                            self.trigger_exception(Exception::AddressLoad, current_pc, Some(vaddr));
                        } else {
                             match bus.read_u16(paddr) {
                                Ok(halfword) => {
                                    let value = halfword as u64; // Zero-extend halfword
                                    self.write_gpr(rt, value);
                                    self.next_pc = self.pc.wrapping_add(4);
                                }
                                Err(_err) => { self.trigger_exception(Exception::BusLoadStore, current_pc, None); }
                            }
                        }
                    }
                     Err(exception) => { self.trigger_exception(exception, current_pc, Some(vaddr)); }
                 }
            },
            0x27 => { // LWU
                let base = self.read_gpr(rs);
                let offset = (imm as i16) as u64; // Sign-extend offset
                let vaddr = base.wrapping_add(offset);
                match self.translate_vaddr(vaddr, AccessType::DataLoad) {
                    Ok(paddr) => {
                         if vaddr % 4 != 0 {
                            self.trigger_exception(Exception::AddressLoad, current_pc, Some(vaddr));
                        } else {
                             match bus.read_u32(paddr) {
                                Ok(word) => {
                                    let value = word as u64; // Zero-extend word
                                    self.write_gpr(rt, value);
                                    self.next_pc = self.pc.wrapping_add(4);
                                }
                                Err(_err) => { self.trigger_exception(Exception::BusLoadStore, current_pc, None); }
                            }
                        }
                    }
                     Err(exception) => { self.trigger_exception(exception, current_pc, Some(vaddr)); }
                }
            },
            0x28 => { // SB
                let base = self.read_gpr(rs);
                let offset = (imm as i16) as u64; // Sign-extend offset
                let vaddr = base.wrapping_add(offset);
                let value = (self.read_gpr(rt) & 0xFF) as u8; // Lower 8 bits of rt
                match self.translate_vaddr(vaddr, AccessType::DataStore) {
                    Ok(paddr) => {
                         // No alignment check for SB
                        if let Err(_err) = bus.write_u8(paddr, value) {
                            self.trigger_exception(Exception::BusLoadStore, current_pc, None);
                        } else {
                            self.next_pc = self.pc.wrapping_add(4);
                        }
                    }
                    Err(exception) => { self.trigger_exception(exception, current_pc, Some(vaddr)); }
                }
            },
            0x29 => { // SH 
                let base = self.read_gpr(rs);
                let offset = (imm as i16) as u64; // Sign-extend offset
                let vaddr = base.wrapping_add(offset);
                let value = (self.read_gpr(rt) & 0xFFFF) as u16; // Lower 16 bits of rt
                match self.translate_vaddr(vaddr, AccessType::DataStore) {
                    Ok(paddr) => {
                        if vaddr % 2 != 0 {
                            self.trigger_exception(Exception::AddressStore, current_pc, Some(vaddr));
                        } else {
                            if let Err(_err) = bus.write_u16(paddr, value) {
                                self.trigger_exception(Exception::BusLoadStore, current_pc, None);
                            } else {
                                self.next_pc = self.pc.wrapping_add(4);
                            }
                        }
                    }
                     Err(exception) => { self.trigger_exception(exception, current_pc, Some(vaddr)); }
                }
            },
            0x2B => { // SW
                let base = self.read_gpr(rs);
                let offset = (imm as i16) as u64; // Sign-extend offset
                let vaddr = base.wrapping_add(offset);
                let value = (self.read_gpr(rt) & 0xFFFFFFFF) as u32; // Lower 32 bits of rt
                 match self.translate_vaddr(vaddr, AccessType::DataStore) {
                     Ok(paddr) => {
                        if vaddr % 4 != 0 {
                            self.trigger_exception(Exception::AddressStore, current_pc, Some(vaddr));
                         } else {
                            if let Err(_err) = bus.write_u32(paddr, value) {
                                self.trigger_exception(Exception::BusLoadStore, current_pc, None);
                            } else {
                                self.next_pc = self.pc.wrapping_add(4);
                            }
                        }
                     }
                    Err(exception) => { self.trigger_exception(exception, current_pc, Some(vaddr)); }
                }
            },
            0x30 => { // LL 
                let base = self.read_gpr(rs);
                let offset = (imm as i16) as u64; // Sign-extend offset
                let vaddr = base.wrapping_add(offset);
                match self.translate_vaddr(vaddr, AccessType::DataLoad) {
                     Ok(paddr) => {
                         if vaddr % 4 != 0 {
                            self.trigger_exception(Exception::AddressLoad, current_pc, Some(vaddr));
                        } else {
                            match bus.read_u32(paddr) {
                                Ok(word) => {
                                    let value = (word as i32) as u64; // Sign-extend word
                                    self.write_gpr(rt, value);
                                    self.ll_bit = true; // Set Load-Linked bit
                                    self.next_pc = self.pc.wrapping_add(4);
                                }
                                Err(_err) => { 
                                    self.trigger_exception(Exception::BusLoadStore, current_pc, None); 
                                    // Don't set next_pc on exception
                                }
                            }
                        }
                    }
                    Err(exception) => { self.trigger_exception(exception, current_pc, Some(vaddr)); }
                }
            },
            0x31 => { /* LDC1 */ 
                self.next_pc = self.pc.wrapping_add(4);
            },
             0x37 => { // LD
                 let base = self.read_gpr(rs);
                 let offset = (imm as i16) as u64; // Sign-extend offset
                 let vaddr = base.wrapping_add(offset);
                  match self.translate_vaddr(vaddr, AccessType::DataLoad) {
                    Ok(paddr) => {
                        if vaddr % 8 != 0 {
                           self.trigger_exception(Exception::AddressLoad, current_pc, Some(vaddr));
                        } else {
                            match bus.read_u64(paddr) {
                                Ok(dword) => {
                                    self.write_gpr(rt, dword);
                                    self.next_pc = self.pc.wrapping_add(4);
                                }
                                Err(_err) => { self.trigger_exception(Exception::BusLoadStore, current_pc, None); }
                            }
                        }
                    }
                     Err(exception) => { self.trigger_exception(exception, current_pc, Some(vaddr)); }
                 }
            },
            0x38 => { // SC 
                let base = self.read_gpr(rs);
                let offset = (imm as i16) as u64; // Sign-extend offset
                let vaddr = base.wrapping_add(offset);
                let value = (self.read_gpr(rt) & 0xFFFFFFFF) as u32; // Lower 32 bits of rt
                match self.translate_vaddr(vaddr, AccessType::DataStore) {
                    Ok(paddr) => {
                         if vaddr % 4 != 0 {
                            self.trigger_exception(Exception::AddressStore, current_pc, Some(vaddr));
                         } else {
                            if self.ll_bit {
                                // TODO: Check physical address match if caching
                                match bus.write_u32(paddr, value) {
                                    Ok(_) => {
                                        self.write_gpr(rt, 1); // Store succeeded
                                        // next_pc set below
                                    }
                                    Err(_err) => {
                                        self.trigger_exception(Exception::BusLoadStore, current_pc, None);
                                        self.write_gpr(rt, 0); // Store failed
                                        // Don't set next_pc on exception
                                        return; // Stop processing after exception
                                    }
                                }
                                self.ll_bit = false; 
                            } else {
                                self.write_gpr(rt, 0); // Store failed because ll_bit was false
                            }
                             self.next_pc = self.pc.wrapping_add(4);
                        }
                    }
                    Err(exception) => { self.trigger_exception(exception, current_pc, Some(vaddr)); }
                }
            },
            0x39 => { /* SDC1 */ 
                self.next_pc = self.pc.wrapping_add(4);
            },
            0x3F => { // SD
                let base = self.read_gpr(rs);
                let offset = (imm as i16) as u64; // Sign-extend offset
                let vaddr = base.wrapping_add(offset);
                let value = self.read_gpr(rt);
                 match self.translate_vaddr(vaddr, AccessType::DataStore) {
                    Ok(paddr) => {
                         if vaddr % 8 != 0 {
                            self.trigger_exception(Exception::AddressStore, current_pc, Some(vaddr));
                        } else {
                            if let Err(_err) = bus.write_u64(paddr, value) {
                                self.trigger_exception(Exception::BusLoadStore, current_pc, None);
                            } else {
                                self.next_pc = self.pc.wrapping_add(4);
                            }
                        }
                    }
                     Err(exception) => { self.trigger_exception(exception, current_pc, Some(vaddr)); }
                }
            },
            // Catch-all for unimplemented/invalid primary opcodes
            _ => { 
                // Unimplemented primary opcode
                 self.trigger_exception(Exception::ReservedInstruction, current_pc, None);
            }
        }
    }

    /// Translates a virtual address to a physical address using the TLB or direct mapping.
    /// Returns the physical address or an Exception cause if translation fails.
    fn translate_vaddr(&mut self, vaddr: u64, access_type: AccessType) -> Result<u64, Exception> {
        // TODO: Check Status.ERL / EXL for bypass? Assume off for now.
        // Check MIPS Architecture modes (User, Supervisor, Kernel)
        // Status Register: KUc (current), KSU (bits 4:3), EXL (bit 1), ERL (bit 2)
        let status = self.cp0.status;
        let is_kernel = (status >> 3) & 0b11 == 0b00 || (status >> 1) & 1 == 1 || (status >> 2) & 1 == 1;
        let is_supervisor = (status >> 3) & 0b11 == 0b01 && !((status >> 1) & 1 == 1 || (status >> 2) & 1 == 1);
        let is_user = (status >> 3) & 0b11 == 0b10 && !((status >> 1) & 1 == 1 || (status >> 2) & 1 == 1);

        // MIPS Virtual Memory Map Segments (based on R4300i User Manual, Ch 5)
        match (vaddr >> 29) & 0b111 { // Check VA[31:29] for 32-bit compatibility mode address spaces
            0b000..=0b011 => { // kuseg (0x0000_0000 - 0x7FFF_FFFF): User, TLB Mapped
                if !is_user {
                    println!("Address Error: KUSEG access outside User mode (VA: {:#x})", vaddr);
                    return Err(if access_type == AccessType::DataStore { Exception::AddressStore }
                           else { Exception::AddressLoad }); // Use AddressLoad for Fetch/Load
                }
                // Proceed to TLB lookup
            }
            0b100 => { // kseg0 (0x8000_0000 - 0x9FFF_FFFF): Kernel, Direct Map, Cached
                if !is_kernel {
                    println!("Address Error: KSEG0 access outside Kernel mode (VA: {:#x})", vaddr);
                     return Err(if access_type == AccessType::DataStore { Exception::AddressStore }
                           else { Exception::AddressLoad }); // Use AddressLoad for Fetch/Load
                }
                let paddr = vaddr & 0x1FFF_FFFF; // Physical = vaddr[28:0]
                // TODO: Handle cache attribute
                return Ok(paddr);
            }
            0b101 => { // kseg1 (0xA000_0000 - 0xBFFF_FFFF): Kernel, Direct Map, Uncached
                 if !is_kernel {
                    println!("Address Error: KSEG1 access outside Kernel mode (VA: {:#x})", vaddr);
                    return Err(if access_type == AccessType::DataStore { Exception::AddressStore }
                          else { Exception::AddressLoad }); // Use AddressLoad for Fetch/Load
                 }
                 let paddr = vaddr & 0x1FFF_FFFF; // Physical = vaddr[28:0]
                 // TODO: Handle cache attribute (uncached)
                 return Ok(paddr);
            }
            0b110 => { // ksseg (0xC000_0000 - 0xDFFF_FFFF): Supervisor, TLB Mapped
                 if !is_supervisor && !is_kernel { // Kernel mode can also access ksseg
                     println!("Address Error: KSSSEG access outside Sup/Kernel mode (VA: {:#x})", vaddr);
                     return Err(if access_type == AccessType::DataStore { Exception::AddressStore }
                           else { Exception::AddressLoad }); // Use AddressLoad for Fetch/Load
                 }
                 // Proceed to TLB lookup
            }
            0b111 => { // kseg3 (0xE000_0000 - 0xFFFF_FFFF): Kernel, TLB Mapped
                 if !is_kernel {
                     println!("Address Error: KSEG3 access outside Kernel mode (VA: {:#x})", vaddr);
                    return Err(if access_type == AccessType::DataStore { Exception::AddressStore }
                           else { Exception::AddressLoad }); // Use AddressLoad for Fetch/Load
                 }
                 // Proceed to TLB lookup
            }
            // Handle 64-bit address spaces if needed (xkphys, xkseg) later
            // For now, assume only 32-bit compatible addresses are used.
             _ => {
                 // This case should ideally not be hit if using 32-bit compatible addresses
                 println!("Address Error: Unhandled address space (VA: {:#x})", vaddr);
                 return Err(if access_type == AccessType::DataStore { Exception::AddressStore }
                        else { Exception::AddressLoad }); // Use AddressLoad for Fetch/Load
             }
        }

        // TLB Lookup for mapped segments (kuseg, ksseg, kseg3)
        println!("TLB Lookup required for VA: {:#018x}", vaddr);

        // Extract ASID from EntryHi
        let current_asid = (self.cp0.entry_hi & 0xFF) as u8;

        for (index, entry) in self.cp0.tlb.iter().enumerate() {
             // Calculate masks based on PageMask register for this entry
             let page_mask_bits = entry.page_mask_raw(); // PageMask[24:13]
             let vpn_comparison_mask = !((page_mask_bits | 0x1FFF) as u64); // Mask for VA[63:13+N]
             let page_size_mask = (page_mask_bits | 0x1FFF) as u64; // Mask for VA[12+N:0] (offset)

             // Compare masked VPN2 (VA[63:13+N]) and ASID (or Global bit)
             let tlb_vpn2 = entry.vpn2(); // EntryHi[63:13] & !0xFF
             if (vaddr & vpn_comparison_mask) == (tlb_vpn2 & vpn_comparison_mask) {
                 // VPN matches for this page size
                 if entry.is_global() || entry.asid() == current_asid {
                    // ASID matches or entry is global

                    // Determine which page entry (Lo0 or Lo1) based on VA bit 12 + N (where N depends on PageMask)
                    // VA[12] selects between the pair.
                    let _page_select_bit_pos = 12 + (page_mask_bits.trailing_zeros() - 13 + 1); // Prefixed unused var
                    // Correction: Bit 12 always selects between the pair, regardless of page size.
                    // The PageMask determines *which* pair of pages the VA falls into.
                    let page_index = ((vaddr >> 12) & 1) as usize;


                    if !entry.is_valid(page_index) {
                         println!("TLB Invalid fault: VA={:#x}, Entry={}, Page={}", vaddr, index, page_index);
                         // Set registers for exception handler
                         self.update_tlb_miss_context(vaddr, current_asid);
                        return Err(if access_type == AccessType::DataStore { Exception::TLBStore } else { Exception::TLBLoad });
                    }

                    if access_type == AccessType::DataStore && !entry.is_dirty(page_index) {
                        println!("TLB Modification fault: VA={:#x}, Entry={}, Page={}", vaddr, index, page_index);
                        // Set registers for exception handler
                         self.update_tlb_miss_context(vaddr, current_asid);
                        return Err(Exception::TLBModification);
                    }

                    // --- TLB Hit ---
                    let pfn = entry.pfn(page_index); // PFN = EntryLo[35:6]
                    let page_offset = vaddr & page_size_mask; // Offset = VA[12+N:0]

                    // Physical address = (PFN << 6) | page_offset
                    // MIPS R4300i Manual Fig 5-16: PA[31:12+N+1] = PFN[35:6+N+1], PA[12+N:0] = VA[12+N:0]
                    // This implies shifting PFN left by 6.
                    let paddr = (pfn << 6) | page_offset;

                    println!("TLB Hit: VA={:#x} -> PA={:#x} (Entry {}, Page {})", vaddr, paddr, index, page_index);

                    // TODO: Handle Cache Coherency attribute entry.cache_coherency(page_index)
                     return Ok(paddr);
                }
            }
        }

        // --- TLB Miss ---
        println!("TLB Miss for VA: {:#018x}", vaddr);
        // Set registers for exception handler
        self.update_tlb_miss_context(vaddr, current_asid);
        Err(if access_type == AccessType::DataStore { Exception::TLBStore } else { Exception::TLBLoad })
    }

    /// Helper to update CP0 context registers on a TLB miss/invalid/modification exception
    fn update_tlb_miss_context(&mut self, vaddr: u64, current_asid: u8) {
        // BadVAddr is set by trigger_exception based on the exception type

        // Context Register: Set BadVPN2 field
        // BadVPN2 = VA[63:13]
        let bad_vpn2 = vaddr >> 13;
        // Context[63:23] = BadVPN2[40:0] (sign extended?) - MIPS R4300i has 40-bit VPN2
        // Assuming direct mapping for now Context[63:23] = VA[63:23]
        self.cp0.context = (self.cp0.context & 0x7FFFFF) | (bad_vpn2 << 23); // Keep PTEBase, set BadVPN2

        // XContext Register: Set BadVPN2 and R fields
        // BadVPN2 = VA[63:13]
        // R = VA[63:62]
        let region = (vaddr >> 62) & 0b11;
        // XContext[63:31] = BadVPN2[31:0] (lower part)
        // XContext[3:1] = R
        self.cp0.xcontext = (self.cp0.xcontext & 0x1) | // Keep PTEBase[39:32]
                            ((bad_vpn2 & 0xFFFFFFFF) << 31) | // Set BadVPN2[31:0]
                            (region << 1); // Set R

        // EntryHi Register: Set VPN2 and ASID
        // VPN2 = VA[63:13]
        // ASID = Current ASID
        self.cp0.entry_hi = (vaddr & 0xFFFFFFFFFFFFF000) | (current_asid as u64);
    }
}

/// Extracts the primary 6-bit opcode (bits 31-26) from an instruction word.
#[inline(always)]
fn opcode(instruction: u32) -> u8 {
    (instruction >> 26) as u8
}

/// Extracts the 5-bit `rs` register index (bits 25-21).
#[inline(always)]
fn rs(instruction: u32) -> u8 {
    ((instruction >> 21) & 0x1F) as u8
}

/// Extracts the 5-bit `rt` register index (bits 20-16).
#[inline(always)]
fn rt(instruction: u32) -> u8 {
    ((instruction >> 16) & 0x1F) as u8
}

/// Extracts the 5-bit `rd` register index (bits 15-11).
#[inline(always)]
fn rd(instruction: u32) -> u8 {
    ((instruction >> 11) & 0x1F) as u8
}

/// Extracts the 5-bit shift amount `shamt` (bits 10-6).
#[inline(always)]
fn shamt(instruction: u32) -> u8 {
    ((instruction >> 6) & 0x1F) as u8
}

/// Extracts the 6-bit function code `funct` (bits 5-0).
#[inline(always)]
fn funct(instruction: u32) -> u8 {
    (instruction & 0x3F) as u8
}

/// Extracts the 16-bit immediate value `imm` (bits 15-0).
#[inline(always)]
fn imm(instruction: u32) -> u16 {
    (instruction & 0xFFFF) as u16
}

/// Extracts the 26-bit jump target `target` (bits 25-0).
#[inline(always)]
fn target(instruction: u32) -> u32 {
    instruction & 0x03FF_FFFF
}

// Basic tests can go here or in a separate tests/ module
#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::Bus;

    // A simple mock bus for testing CPU initialization and basic operations
    struct MockBus;

    impl Bus for MockBus {
        fn read_u8(&mut self, _vaddr: u64) -> Result<u8, crate::bus::BusError> { Err(crate::bus::BusError::AddressError) }
        fn read_u16(&mut self, _vaddr: u64) -> Result<u16, crate::bus::BusError> { Err(crate::bus::BusError::AddressError) }
        fn read_u32(&mut self, _vaddr: u64) -> Result<u32, crate::bus::BusError> { Err(crate::bus::BusError::AddressError) }
        fn read_u64(&mut self, _vaddr: u64) -> Result<u64, crate::bus::BusError> { Err(crate::bus::BusError::AddressError) }
        fn write_u8(&mut self, _vaddr: u64, _value: u8) -> Result<(), crate::bus::BusError> { Err(crate::bus::BusError::AddressError) }
        fn write_u16(&mut self, _vaddr: u64, _value: u16) -> Result<(), crate::bus::BusError> { Err(crate::bus::BusError::AddressError) }
        fn write_u32(&mut self, _vaddr: u64, _value: u32) -> Result<(), crate::bus::BusError> { Err(crate::bus::BusError::AddressError) }
        fn write_u64(&mut self, _vaddr: u64, _value: u64) -> Result<(), crate::bus::BusError> { Err(crate::bus::BusError::AddressError) }
    }

    #[test]
    fn cpu_initial_state() {
        let cpu = Cpu::new();
        assert_eq!(cpu.pc, 0xFFFFFFFF_BFC0_0000);
        assert_eq!(cpu.gpr[0], 0);
        assert_eq!(cpu.hi, 0);
        assert_eq!(cpu.lo, 0);
        assert!(cpu.ll_bit == false);
        assert_eq!(cpu.cp0.status, 0x34000000);
        assert_eq!(cpu.cp0.cause, 0);
        assert_eq!(cpu.cp0.epc, 0);
        assert_eq!(cpu.cp0.prid, 0x0B00);
        assert_eq!(cpu.cp0.config, 0x7006E460);
        assert_eq!(cpu.cp0.badvaddr, 0);
        assert_eq!(cpu.cp1.fir, 0x00001400);
        assert_eq!(cpu.cp1.fcsr, 0x01000000);
        // Add more checks as state gets added
    }

    // TODO: Add tests for step function once fetch/decode/execute is implemented
    // Example:
    // #[test]
    // fn test_nop_instruction() {
    //     let mut cpu = Cpu::new();
    //     let mut bus = MockBus; // Needs a way to provide instructions
    //     // Setup bus memory to return a NOP instruction (0x00000000) at RESET_VECTOR
    //     // cpu.step(&mut bus);
    //     // Assert PC advanced by 4, registers unchanged, etc.
    // }
} 
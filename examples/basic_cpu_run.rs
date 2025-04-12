// examples/basic_cpu_run.rs

// Use the snake_case version of the crate name
use n64_emulator_rust::cpu::Cpu;
use n64_emulator_rust::bus::{Bus, SimpleBus};

fn main() {
    println!("--- Basic CPU Run Example ---");

    // N64 Physical Memory Size (4MB)
    const RAM_SIZE: usize = 4 * 1024 * 1024;
    // N64 Boot ROM physical address (where the CPU starts fetching)
    // const N64_BOOT_ROM_PADDR: u64 = 0x1FC0_0000; // Physical address for 0xBFC00000
    // For SimpleBus test, load at the start of RAM
    const TEST_LOAD_PADDR: u64 = 0x0; 
    const TEST_LOAD_VADDR: u64 = 0xFFFFFFFF_80000000; // KSEG0 address mapping to 0x0

    let mut bus = SimpleBus::new(RAM_SIZE);
    let mut cpu = Cpu::new();
    cpu.set_pc(TEST_LOAD_VADDR); // Override the reset vector for this test

    // Let's write a simple program into the RAM at the boot address
    // Program:
    // 1. LUI t0, 0x1234         (t0 = 0x12340000)
    // 2. ORI t0, t0, 0x5678    (t0 = 0x12345678)
    // 3. DADDIU t1, t0, 1      (t1 = t0 + 1)
    // 4. NOP                   (Infinite loop placeholder)

    let instructions: [u32; 4] = [
        0x3C081234, // LUI t0 ($8), 0x1234
        0x35085678, // ORI t0 ($8), t0 ($8), 0x5678
        0x65090001, // DADDIU t1 ($9), t0 ($8), 1
        0x00000000, // NOP (effectively SLL r0, r0, 0)
    ];

    println!("Writing instructions to physical address {:#x}...", TEST_LOAD_PADDR);
    for (i, instruction) in instructions.iter().enumerate() {
        let paddr = TEST_LOAD_PADDR + (i * 4) as u64;
        match bus.write_u32(paddr, *instruction) {
            Ok(_) => println!("  Wrote {:#010x} to {:#x}", instruction, paddr),
            Err(e) => {
                eprintln!("Error writing instruction to bus: {:?}", e);
                return;
            }
        }
    }

    // Check initial state
    // Note: We need a way to read GPRs for verification. Let's add a helper to Cpu.
    // For now, we just observe the PC and any prints from cpu.step()
    println!("\nInitial CPU PC: {:#018x}", cpu.pc()); // Should be TEST_LOAD_VADDR

    // Run a few steps
    let num_steps = 5; // Execute LUI, ORI, DADDIU, NOP, and the NOP again
    println!("\nRunning {} CPU steps...", num_steps);
    for i in 0..num_steps {
        println!("\n--- Step {} ---", i);
        cpu.step(&mut bus);
        // Ideally, we'd print register state here if Cpu had a public method for it.
        println!("Current CPU PC: {:#018x}", cpu.pc()); // Shows PC for the *next* instruction fetch
        println!("Current next_pc: {:#018x}", cpu.next_pc()); // Shows where the PC will go after *this* instruction executes
    }

    println!("\n--- Finished ---");
    // TODO: Add assertions once we can inspect GPRs
    // e.g., assert_eq!(cpu.read_gpr_debug(8), 0x12345678);
    //       assert_eq!(cpu.read_gpr_debug(9), 0x12345679);
    let gpr8_val = cpu.read_gpr_debug(8); // t0
    let gpr9_val = cpu.read_gpr_debug(9); // t1
    println!("Final GPR $t0 (8): {:#018x}", gpr8_val);
    println!("Final GPR $t1 (9): {:#018x}", gpr9_val);

    // Add assertions for verification
    assert_eq!(gpr8_val, 0x12345678, "GPR $t0 ($8) should be 0x12345678");
    assert_eq!(gpr9_val, 0x12345679, "GPR $t1 ($9) should be 0x12345679");

    println!("\nAssertions passed!");
} 
#![allow(dead_code)] // TODO: Remove later

/// Represents the different causes for CPU exceptions.
/// Values correspond to the MIPS `Cause` register ExcCode field (bits 6:2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Exception {
    /// Interrupt (external, timer, etc.)
    Interrupt = 0,
    /// TLB Modification exception
    TLBModification = 1,
    /// TLB Load exception (including Fetch)
    TLBLoad = 2,
    /// TLB Store exception
    TLBStore = 3,
    /// Address Error on Load (including Fetch)
    AddressLoad = 4,
    /// Address Error on Store
    AddressStore = 5,
    /// Bus Error on Instruction Fetch
    BusFetch = 6,
    /// Bus Error on Load/Store
    BusLoadStore = 7,
    /// Syscall instruction
    Syscall = 8,
    /// Breakpoint instruction
    Breakpoint = 9,
    /// Reserved Instruction
    ReservedInstruction = 10,
    /// Coprocessor Unusable
    CoprocessorUnusable = 11,
    /// Arithmetic Overflow
    Overflow = 12,
    /// Trap instruction
    Trap = 13,
    // 14 Reserved
    /// Floating Point Exception
    FloatingPoint = 15,
    // 16-22 Reserved
    /// Watchpoint exception
    Watch = 23,
    // 24-31 Reserved
} 
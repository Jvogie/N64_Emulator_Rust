#![allow(unused_variables)] // TODO: Remove this when reads/writes are implemented

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusError {
    /// Access to an address that is not mapped or invalid.
    AddressError,
    // TODO: Add other potential bus errors (e.g., AlignmentError, TLB exceptions?)
}

/// Represents the system bus, allowing the CPU to interact with memory and peripherals.
pub trait Bus {
    /// Reads an 8-bit unsigned byte from the specified virtual address.
    fn read_u8(&mut self, vaddr: u64) -> Result<u8, BusError>;
    /// Reads a 16-bit unsigned half-word from the specified virtual address.
    fn read_u16(&mut self, vaddr: u64) -> Result<u16, BusError>;
    /// Reads a 32-bit unsigned word from the specified virtual address.
    fn read_u32(&mut self, vaddr: u64) -> Result<u32, BusError>;
    /// Reads a 64-bit unsigned double-word from the specified virtual address.
    fn read_u64(&mut self, vaddr: u64) -> Result<u64, BusError>;

    /// Writes an 8-bit unsigned byte to the specified virtual address.
    fn write_u8(&mut self, vaddr: u64, value: u8) -> Result<(), BusError>;
    /// Writes a 16-bit unsigned half-word to the specified virtual address.
    fn write_u16(&mut self, vaddr: u64, value: u16) -> Result<(), BusError>;
    /// Writes a 32-bit unsigned word to the specified virtual address.
    fn write_u32(&mut self, vaddr: u64, value: u32) -> Result<(), BusError>;
    /// Writes a 64-bit unsigned double-word to the specified virtual address.
    fn write_u64(&mut self, vaddr: u64, value: u64) -> Result<(), BusError>;
}

/// A simple bus implementation with a single block of RAM.
pub struct SimpleBus {
    ram: Vec<u8>,
}

impl SimpleBus {
    /// Creates a new SimpleBus with the specified RAM size in bytes.
    pub fn new(size: usize) -> Self {
        SimpleBus { ram: vec![0; size] }
    }
}

impl Bus for SimpleBus {
    fn read_u8(&mut self, paddr: u64) -> Result<u8, BusError> {
        let addr = paddr as usize;
        if addr < self.ram.len() {
            Ok(self.ram[addr])
        } else {
            Err(BusError::AddressError)
        }
    }

    fn read_u16(&mut self, paddr: u64) -> Result<u16, BusError> {
        let addr = paddr as usize;
        if addr + 1 < self.ram.len() {
            // N64 is big-endian
            let bytes = [self.ram[addr], self.ram[addr + 1]];
            Ok(u16::from_be_bytes(bytes))
        } else {
            Err(BusError::AddressError)
        }
    }

    fn read_u32(&mut self, paddr: u64) -> Result<u32, BusError> {
        let addr = paddr as usize;
        if addr + 3 < self.ram.len() {
            // N64 is big-endian
            let bytes = [self.ram[addr], self.ram[addr + 1], self.ram[addr + 2], self.ram[addr + 3]];
            Ok(u32::from_be_bytes(bytes))
        } else {
            Err(BusError::AddressError)
        }
    }

    fn read_u64(&mut self, paddr: u64) -> Result<u64, BusError> {
        let addr = paddr as usize;
        if addr + 7 < self.ram.len() {
            // N64 is big-endian
            let bytes = [self.ram[addr],     self.ram[addr + 1], self.ram[addr + 2], self.ram[addr + 3],
                         self.ram[addr + 4], self.ram[addr + 5], self.ram[addr + 6], self.ram[addr + 7]];
            Ok(u64::from_be_bytes(bytes))
        } else {
            Err(BusError::AddressError)
        }
    }

    fn write_u8(&mut self, paddr: u64, value: u8) -> Result<(), BusError> {
        let addr = paddr as usize;
        if addr < self.ram.len() {
            self.ram[addr] = value;
            Ok(())
        } else {
            Err(BusError::AddressError)
        }
    }

    fn write_u16(&mut self, paddr: u64, value: u16) -> Result<(), BusError> {
        let addr = paddr as usize;
        if addr + 1 < self.ram.len() {
            // N64 is big-endian
            let bytes = value.to_be_bytes();
            self.ram[addr] = bytes[0];
            self.ram[addr + 1] = bytes[1];
            Ok(())
        } else {
            Err(BusError::AddressError)
        }
    }

    fn write_u32(&mut self, paddr: u64, value: u32) -> Result<(), BusError> {
        let addr = paddr as usize;
        if addr + 3 < self.ram.len() {
            // N64 is big-endian
            let bytes = value.to_be_bytes();
            self.ram[addr] = bytes[0];
            self.ram[addr + 1] = bytes[1];
            self.ram[addr + 2] = bytes[2];
            self.ram[addr + 3] = bytes[3];
            Ok(())
        } else {
            Err(BusError::AddressError)
        }
    }

    fn write_u64(&mut self, paddr: u64, value: u64) -> Result<(), BusError> {
        let addr = paddr as usize;
        if addr + 7 < self.ram.len() {
            // N64 is big-endian
            let bytes = value.to_be_bytes();
            self.ram[addr] = bytes[0];
            self.ram[addr + 1] = bytes[1];
            self.ram[addr + 2] = bytes[2];
            self.ram[addr + 3] = bytes[3];
            self.ram[addr + 4] = bytes[4];
            self.ram[addr + 5] = bytes[5];
            self.ram[addr + 6] = bytes[6];
            self.ram[addr + 7] = bytes[7];
            Ok(())
        } else {
            Err(BusError::AddressError)
        }
    }
} 
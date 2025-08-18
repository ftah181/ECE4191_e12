import spidev

# Initialize SPI
spi = spidev.SpiDev()
spi.open(0, 0)      # Open SPI bus 0, device (CS) 0
spi.max_speed_hz = 1350000  # MCP3008 works up to ~3.6 MHz

def read_adc(channel: int) -> int:
    """
    Read a value from the specified ADC channel (0-7) on the MCP3008.

    Args:
        channel (int): ADC channel number (0–7)

    Returns:
        int: 10-bit ADC value (0–1023)
    """

    if not (0 <= channel <= 7):
        raise ValueError("Channel must be between 0 and 7")

    # MCP3008 expects 3 bytes:
    # 1. Start bit = 1
    # 2. Single-ended/differential + channel selection
    # 3. Don't care, just clock out zeros
    
    cmd = [1, (8 + channel) << 4, 0]
    reply = spi.xfer2(cmd)  # Perform SPI transaction
    
    # Construct 10-bit result
    result = ((reply[1] & 3) << 8) | reply[2]
    return result

if __name__ == "__main__":
    try:
        while True:
            value = read_adc(0)  # Read channel 0
            print(f"ADC Channel 0 Value: {value}")
    except KeyboardInterrupt:
        spi.close()

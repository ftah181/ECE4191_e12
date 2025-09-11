#-- -- -- -- -- -- -- -- -- Code Start -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
#include <bcm2835.h>
#include <stdint.h>
#include <time.h>
#include <stdio.h>

#define CS_PIN RPI_V2_GPIO_P1_40 // GPIO21

// Read MCP3008 channel (0-7)
uint16_t read_mcp3008(uint8_t channel)
{
    if (channel > 7)
        return 0;

    char buf[3];
    buf[0] = 1;
    buf[1] = (8 + channel) << 4;
    buf[2] = 0;

    bcm2835_gpio_write(CS_PIN, LOW);
    bcm2835_spi_transfern(buf, 3);
    bcm2835_gpio_write(CS_PIN, HIGH);

    uint16_t value = ((buf[1] & 3) << 8) | buf[2];
    return value;
}

// Exposed function for Python: samples MCP3008 into pre-allocated array
double sample_mcp3008(int channel, int num_samples, uint16_t *out_samples)
{
    if (!bcm2835_init())
        return -1;
    if (!bcm2835_spi_begin())
        return -1;

    bcm2835_spi_setBitOrder(BCM2835_SPI_BIT_ORDER_MSBFIRST);
    bcm2835_spi_setDataMode(BCM2835_SPI_MODE0);
    bcm2835_spi_setClockDivider(BCM2835_SPI_CLOCK_DIVIDER_64); // ~1 MHz
    bcm2835_gpio_fsel(CS_PIN, BCM2835_GPIO_FSEL_OUTP);
    bcm2835_gpio_write(CS_PIN, HIGH);

    const double SAMPLE_RATE = 20000.0; // Hz
    const double INTERVAL = 1.0 / SAMPLE_RATE;

    struct timespec t_start, t_now;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for (int i = 0; i < num_samples; i++)
    {
        out_samples[i] = read_mcp3008(channel);

        // Busy wait for precise timing
        do
        {
            clock_gettime(CLOCK_MONOTONIC, &t_now);
        } while (((t_now.tv_sec - t_start.tv_sec) + (t_now.tv_nsec - t_start.tv_nsec) / 1e9) < (i + 1) * INTERVAL);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_now);
    double total_time = (t_now.tv_sec - t_start.tv_sec) + (t_now.tv_nsec - t_start.tv_nsec) / 1e9;

    bcm2835_spi_end();
    bcm2835_close();

    return total_time;
}

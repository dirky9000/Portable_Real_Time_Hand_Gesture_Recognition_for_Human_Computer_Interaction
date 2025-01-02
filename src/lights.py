from gpiozero import LED
from time import sleep

red = LED(22)
green = LED(27)
blue = LED(17)

while True:
    red.on()
    sleep(1)
    red.off()
    green.on()
    sleep(1)
    green.off()
    blue.on()
    sleep(1)
    blue.off()
from ev3dev2.wheel import Wheel

class catterpillerWheel(Wheel):
    def __init__(self):
        Wheel.__init__(self,31,5,45)
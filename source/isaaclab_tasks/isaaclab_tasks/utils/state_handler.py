from typing import TypeVar, Generic
from enum import Enum
import warp as wp
import time


# generic state handler 
T=TypeVar('T')
@wp.struct
class StateHandler:
    @wp.kernel
    def __init__(self, name: str):
        self.stateType = name
        self.currentState:  wp.Int =0
        self.pastState = wp.Int =0
        self.stateChange : bool  = self._stateChange()

    @wp.func
    def _stateChange(self):
        # return true if the states have changed
        return (self.pastState!=self.currentState)
    
    @wp.func
    def printState(self):
        # print the value of the state 
        print(f"[INFO] : {self.stateType} : {self.currentState}")
        
    @wp.func
    def handleChange(self, newstate ):
        self.currentState = newstate
        if self._stateChange():
            self.pastState = newstate
            self.printState()


# class PickSmState:
#     """States for the pick state machine."""
#     REST = wp.constant(0)
#     APPROACH_ABOVE_OBJECT = wp.constant(1)
#     APPROACH_OBJECT = wp.constant(2)
#     GRASP_OBJECT = wp.constant(3)
#     LIFT_OBJECT = wp.constant(4)
    

# wp.init()
# state = PickSmState.REST

# stateHandler = StateHandler(PickSmState, "PickSmState")
# print("start")
# time.sleep(1)
# print("setting state...")
# time.sleep(1)
# state=PickSmState.APPROACH_ABOVE_OBJECT
# stateHandler.handleChange(state)
# time.sleep(1)
# state=PickSmState.APPROACH_ABOVE_OBJECT
# stateHandler.handleChange(state)
# time.sleep(1)
# state=PickSmState.GRASP_OBJECT
# stateHandler.handleChange(state)

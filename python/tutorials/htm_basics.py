import numpy as np
from htm.bindings.sdr import SDR
# Uncomment to get info
#import htm.bindings.sdr
#help(htm.bindings.sdr.SDR)
#help(htm.bindings.algorithms.TemporalMemory)
from htm.algorithms import TemporalMemory as TM

"""
Based on this fabtastic tutorial:
https://3rdman.de/2020/02/hierarchical-temporal-memory-part-1-getting-started/
"""

# 1. Create a semantic representation of the integer numbers from 0 to 9

arraySize = 80
cycleArray = np.arange(0, 10, 1)
print(cycleArray)
inputSDR = SDR( arraySize )

# 2. Turn this representation into an SDR / ENCONDING
# Another thing to be aware of is, that the Temporal Memory actually needs at list 8 active bits, to work as expected.
def formatSdr(sdr):

  result = ''
  for i in range(sdr.size):
    if i > 0 and i % 8 == 0:
      result += ' '
    result += str(sdr.dense.flatten()[i])
  return result

# 3. Create a temporal memory and train it with the SDR
tm = TM(columnDimensions = (inputSDR.size,),
        cellsPerColumn = 1,       # default: 32
        minThreshold = 4,         # default: 10
        activationThreshold = 8,  # default: 13
        initialPermanence = 0.5,  # default: 0.21
        )

for cycle in range(2):
    for sensorValue in cycleArray:
        sensorValueBits = inputSDR.dense
        sensorValueBits = np.zeros(arraySize)
        sensorValueBits[sensorValue * 8:sensorValue * 8 + 8] = 1
        inputSDR.dense = sensorValueBits
        tm.compute(inputSDR, learn = True)
        print(format(sensorValue,'>2') + '/' + format(cycle, '1d')+ ' |', formatSdr(tm.getActiveCells()), 'Active')

        tm.activateDendrites(True)
        print(format(tm.anomaly, '.2f') + ' |', formatSdr(tm.getPredictiveCells()), 'Predicted')


# 4. Have a look at the predictions of the TM after training with another range:
cycleArray2 = np.arange(0, 10, 1)
# We add an anomaly to the secuence
cycleArray2[2]=3

print("Anomally prediction.....")
for sensorValue in cycleArray2:
    sensorValueBits = inputSDR.dense
    sensorValueBits = np.zeros(arraySize)
    sensorValueBits[sensorValue * 8:sensorValue * 8 + 8] = 1
    inputSDR.dense = sensorValueBits
    tm.compute(inputSDR, learn = True)
    print(format(sensorValue,'>2') + '/' + format(cycle, '1d')+ ' |', formatSdr(tm.getActiveCells()), 'Active')

    tm.activateDendrites(True)
    print(format(tm.anomaly, '.2f') + ' |', formatSdr(tm.getPredictiveCells()), 'Predicted')

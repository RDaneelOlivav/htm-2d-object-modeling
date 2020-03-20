import numpy as np
from htm.bindings.sdr import SDR
# Uncomment to get info
#import htm.bindings.sdr
#help(htm.bindings.sdr.SDR)
#help(htm.bindings.algorithms.TemporalMemory)
#help(htm.bindings.algorithms.SpatialPooler)
from htm.algorithms import TemporalMemory as TM

"""
Based on this fabtastic tutorial:
https://3rdman.de/2020/02/hierarchical-temporal-memory-part-1-getting-started/
"""


from pandaComm.pandaServer import PandaServer
from pandaComm.dataExchange import ServerData, dataHTMObject, dataLayer, dataInput


class HtmBasicExample(object):

    def __init__(self):

        # 1. Create a semantic representation of the integer numbers from 0 to 9
        # We want to represent 10 values with 4 bits each
        self.arraySize = 80
        self.cycleArray = np.arange(0, 10, 1)
        print(self.cycleArray)
        self.inputSDR = SDR( self.arraySize )

        # 3. Create a temporal memory and train it with the SDR
        self.init_model_parameters()
        self.tm = TM(columnDimensions = (self.model_parameters["tm"]["columnDimensions"],),
                cellsPerColumn = self.model_parameters["tm"]["cellsPerColumn"],       # default: 32
                minThreshold = self.model_parameters["tm"]["minThreshold"],         # default: 10
                activationThreshold = self.model_parameters["tm"]["activationThreshold"],  # default: 13
                initialPermanence = self.model_parameters["tm"]["initialPermanence"],  # default: 0.21
                )

        self.init_htmvis()

    def init_model_parameters(self):
        """
        We load the model parameters. Here we will set all the parameters for sp,tm, etc.
        """
        self.model_parameters = {
            'tm': {'columnDimensions': self.inputSDR.size,
                    'cellsPerColumn': 1,
                    'minThreshold': 4,
                    'activationThreshold': 8,
                   'initialPermanence': 0.5},
                   }


    def __del__(self):
        # TODO: I get the error: RuntimeError: cannot join thread before it is started
        self.del_htmvis()


    def init_htmvis(self):

        self.pandaServer = PandaServer() # globally for example
        self.pandaServer.Start()
        self.BuildPandaSystem(columnCount=self.model_parameters["tm"]["columnDimensions"],
                        cellsPerColumn=self.model_parameters["tm"]["cellsPerColumn"])

    def BuildPandaSystem(self, columnCount,cellsPerColumn):

        self.serverData = ServerData()
        self.serverData.HTMObjects["HTM1"] = dataHTMObject()
        self.serverData.HTMObjects["HTM1"].inputs["S_NumberRange"] = dataInput()

        self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"] = dataLayer(columnCount=columnCount,
                                                                        cellsPerColumn=cellsPerColumn)

        self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"].proximalInputs = ["S_NumberRange",]

    def UpdateHtmVisValues(self, sensorValue, sensorSDR, activeCellsSDR,predictiveCellsSDR):

        # ------------------HTMpandaVis----------------------

        # fill up values
        self.serverData.HTMObjects["HTM1"].inputs["S_NumberRange"].stringValue = "sensorValue: {:}".format(sensorValue)
        self.serverData.HTMObjects["HTM1"].inputs["S_NumberRange"].bits = sensorSDR.sparse
        self.serverData.HTMObjects["HTM1"].inputs["S_NumberRange"].count = sensorSDR.size

        self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"].activeColumns = activeCellsSDR.sparse
        self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"].winnerCells = self.tm.getWinnerCells().sparse
        self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"].predictiveCells = predictiveCellsSDR.sparse

        self.pandaServer.serverData = self.serverData

        #pandaServer.spatialPoolers["HTM1"] = sp
        self.pandaServer.temporalMemories["HTM1"] = self.tm
        self.pandaServer.NewStateDataReady()

        print("One step finished")
        while not self.pandaServer.runInLoop and not self.pandaServer.runOneStep:
            pass
        self.pandaServer.runOneStep = False
        print("Proceeding one step...")

        # ------------------HTMpandaVis----------------------

    def del_htmvis(self):

        self.pandaServer.MainThreadQuitted()


    # 2. Turn this representation into an SDR / ENCONDING
    # Another thing to be aware of is, that the Temporal Memory actually needs at list 8 active bits, to work as expected.
    def formatSdr(self, sdr):

        result = ''
        for i in range(sdr.size):
            if i > 0 and i % 8 == 0:
                result += ' '
            result += str(sdr.dense.flatten()[i])
        return result


    def start_training(self):
        """
        We train the TM based on the self.cycleArray
        """

        for cycle in range(2):
            for sensorValue in self.cycleArray:
                sensorValueBits = self.inputSDR.dense
                sensorValueBits = np.zeros(self.arraySize)
                sensorValueBits[sensorValue * 8:sensorValue * 8 + 8] = 1
                self.inputSDR.dense = sensorValueBits
                self.tm.compute(self.inputSDR, learn = True)
                activeCellsSDR = self.tm.getActiveCells()
                print(format(sensorValue,'>2') + '/' + format(cycle, '1d')+ ' |', self.formatSdr(activeCellsSDR), 'Active')

                self.tm.activateDendrites(True)
                predictiveCellsSDR = self.tm.getPredictiveCells()
                print(format(self.tm.anomaly, '.2f') + ' |', self.formatSdr(predictiveCellsSDR), 'Predicted')

                self.UpdateHtmVisValues(sensorValue=sensorValue,
                                        sensorSDR=self.inputSDR,
                                        activeCellsSDR=activeCellsSDR,
                                        predictiveCellsSDR=predictiveCellsSDR)

    def start_test(self):
        # 4. Have a look at the predictions of the TM after training with another range:
        self.cycleArray2 = np.arange(0, 10, 1)
        # We add an anomaly to the secuence
        self.cycleArray2[2]=3

        print("###### Anomally prediction ######")
        cycle = 0
        for sensorValue in self.cycleArray2:
            sensorValueBits = self.inputSDR.dense
            sensorValueBits = np.zeros(self.arraySize)
            sensorValueBits[sensorValue * 8:sensorValue * 8 + 8] = 1
            self.inputSDR.dense = sensorValueBits
            self.tm.compute(self.inputSDR, learn = True)
            activeCellsSDR = self.tm.getActiveCells()
            print(format(sensorValue,'>2') + '/' + format(cycle, '1d')+ ' |', self.formatSdr(activeCellsSDR), 'Active')

            self.tm.activateDendrites(True)
            predictiveCellsSDR = self.tm.getPredictiveCells()
            print(format(self.tm.anomaly, '.2f') + ' |', self.formatSdr(predictiveCellsSDR), 'Predicted')

            self.UpdateHtmVisValues(sensorValue=sensorValue,
                                    sensorSDR=self.inputSDR,
                                    activeCellsSDR=activeCellsSDR,
                                    predictiveCellsSDR=predictiveCellsSDR)


if __name__ == "__main__":
    htmbasic_example_object = HtmBasicExample()
    htmbasic_example_object.start_training()
    htmbasic_example_object.start_test()

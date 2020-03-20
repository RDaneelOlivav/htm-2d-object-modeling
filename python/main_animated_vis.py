#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import yaml
import math
import htm2d.environment
import htm2d.agent
from htm2d.agent import Direction
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import animation

from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters

from pandaComm.pandaServer import PandaServer
from pandaComm.dataExchange import ServerData, dataHTMObject, dataLayer, dataInput


_EXEC_DIR = os.path.dirname(os.path.abspath(__file__))
#go one folder up and then into the objects folder
_OBJECTS_DIR=os.path.join(_EXEC_DIR,os.path.pardir, 'objects')

OBJECT_FILENAME = 'a.yml'#what object to load

class AnimateSDR2D(object):

	def __init__(self, agent_x_init=3, agent_y_init=4):
		# load model parameters from file
		f = open('modelParams.cfg','r').read()
		self.modelParams = eval(f)

		# set up system
		self.SystemSetup(self.modelParams)

		# put agent in the environment
		self.agent.set_env(self.env,agent_x_init,agent_y_init)




	def __del__(self):
		self.del_htmvis()

	def init_htmvis(self, columnCount, cellsPerColumn):

		self.pandaServer = PandaServer() # globally for example
		self.pandaServer.Start()
		self.BuildPandaSystem(columnCount=columnCount,
						cellsPerColumn=cellsPerColumn)

	def BuildPandaSystem(self, columnCount,cellsPerColumn):

		self.serverData = ServerData()
		self.serverData.HTMObjects["HTM1"] = dataHTMObject()
		self.serverData.HTMObjects["HTM1"].inputs["S_ElementSensor"] = dataInput()

		self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"] = dataLayer(columnCount=columnCount,
																		cellsPerColumn=cellsPerColumn)

		self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"].proximalInputs = ["S_ElementSensor",]

	def UpdateHtmVisValues(self, sensorValue, sensorSDR, activeCellsSDR, predictiveCellsSDR, spatial_pooler_object, temporal_pooler_object):

		# ------------------HTMpandaVis----------------------

		# fill up values
		self.serverData.HTMObjects["HTM1"].inputs["S_ElementSensor"].stringValue = "sensorValue: {:}".format(sensorValue)
		self.serverData.HTMObjects["HTM1"].inputs["S_ElementSensor"].bits = sensorSDR.sparse
		self.serverData.HTMObjects["HTM1"].inputs["S_ElementSensor"].count = sensorSDR.size

		self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"].activeColumns = activeCellsSDR.sparse
		self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"].winnerCells = temporal_pooler_object.getWinnerCells().sparse
		self.serverData.HTMObjects["HTM1"].layers["SensoryLayer"].predictiveCells = predictiveCellsSDR.sparse

		self.pandaServer.serverData = self.serverData

		self.pandaServer.spatialPoolers["HTM1"] = spatial_pooler_object
		self.pandaServer.temporalMemories["HTM1"] = temporal_pooler_object
		self.pandaServer.NewStateDataReady()

		print("One step finished")
		while not self.pandaServer.runInLoop and not self.pandaServer.runOneStep:
			pass
		self.pandaServer.runOneStep = False
		print("Proceeding one step...")

		# ------------------HTMpandaVis----------------------

	def del_htmvis(self):

		self.pandaServer.MainThreadQuitted()

	def SystemSetup(self, parameters,verbose=True):
		#global agent, sensorEncoder, env, sensorLayer_sp, sensorLayer_sp_activeColumns

		if verbose:
			import pprint
			print("Parameters:")
			pprint.pprint(parameters, indent=4)
			print("")

		#create environment and the agent
		self.env = htm2d.environment.TwoDimensionalEnvironment(20, 20)
		self.agent = htm2d.agent.Agent()


		#load object from yml file
		with open(os.path.join(_OBJECTS_DIR,OBJECT_FILENAME), 'r') as stream:
			try:
				self.env.load_object(stream)
			except yaml.YAMLError as exc:
				print(exc)

		#SETUP SENSOR ENCODER
		self.sensorEncoderParams            = RDSE_Parameters()
		self.sensorEncoderParams.category   = True
		self.sensorEncoderParams.size       = parameters["enc"]["size"]
		self.sensorEncoderParams.sparsity = parameters["enc"]["sparsity"]
		self.sensorEncoder = RDSE( self.sensorEncoderParams )


		# Make the HTM.  SpatialPooler & TemporalMemory & associated tools.
		spParams = parameters["sensorLayer_sp"]
		self.sensorLayer_sp = SpatialPooler(
			inputDimensions            = (self.sensorEncoder.size,),
			columnDimensions           = (spParams["columnCount"],),
			potentialPct               = spParams["potentialPct"],
			potentialRadius            = self.sensorEncoder.size,
			globalInhibition           = True,
			localAreaDensity           = spParams["localAreaDensity"],
			synPermInactiveDec         = spParams["synPermInactiveDec"],
			synPermActiveInc           = spParams["synPermActiveInc"],
			synPermConnected           = spParams["synPermConnected"],
			boostStrength              = spParams["boostStrength"],
			wrapAround                 = True
		)
		self.sp_info = Metrics(self.sensorLayer_sp.getColumnDimensions(), 999999999 )

		# Create an SDR to represent active columns, This will be populated by the
		# compute method below. It must have the same dimensions as the Spatial Pooler.
		self.sensorLayer_sp_activeColumns = SDR( spParams["columnCount"] )

		tmParams = parameters["tm"]
		self.tm = TemporalMemory(
			columnDimensions          = (spParams["columnCount"],),
			cellsPerColumn            = tmParams["cellsPerColumn"],
			activationThreshold       = tmParams["activationThreshold"],
			initialPermanence         = tmParams["initialPerm"],
			connectedPermanence       = spParams["synPermConnected"],
			minThreshold              = tmParams["minThreshold"],
			maxNewSynapseCount        = tmParams["newSynapseCount"],
			permanenceIncrement       = tmParams["permanenceInc"],
			permanenceDecrement       = tmParams["permanenceDec"],
			predictedSegmentDecrement = 0.0,
			maxSegmentsPerCell        = tmParams["maxSegmentsPerCell"],
			maxSynapsesPerSegment     = tmParams["maxSynapsesPerSegment"]
		)
		self.tm_info = Metrics( [self.tm.numberOfCells()], 999999999 )

		# We initialise the HTMVIS
		self.init_htmvis(columnCount=spParams["columnCount"], cellsPerColumn=tmParams["cellsPerColumn"])

	def SystemCalculate(self, plot=False):
		"""
		plot: We tel if we want a Matplot lib of a HTMvis
		"""
		#global sensorLayer_sp,arr

		# encode sensor data to SDR--------------------------------------------------

		# convert sensed feature to int
		sensedFeature = 1 if self.agent.get_feature(Direction.UP)=='X'else 0

		sensorSDR = self.sensorEncoder.encode(sensedFeature)

		position = self.agent.get_position()
		print("Sensor at {x}, {y}:".format(x=position[0], y=position[1]))
		print("Feature UP: {}".format(sensedFeature))
		print(sensorSDR)

		# put SDR to proximal input of sensorLayer-----------------------------------
		# sensorLayer.proximal = sensorSDR

		# Execute Spatial Pooling algorithm over input space.
		self.sensorLayer_sp.compute(sensorSDR, True, self.sensorLayer_sp_activeColumns)
		self.sp_info.addData(self.sensorLayer_sp_activeColumns)
		activeCellsSDR=self.sensorLayer_sp_activeColumns

		# We compute the TM
		# Execute Temporal Memory algorithm over active mini-columns.
		#self.tm.compute(sensorSDR, learn = True)
		self.tm.activateDendrites(True)
		predictiveCellsSDR = self.tm.getPredictiveCells()

		if plot:
			self.plotBinaryMap("Input SDR", sensorSDR.size, sensorSDR.dense, subplot=121)
			self.plotBinaryMap("Sensor layer columns activation", self.sensorLayer_sp.getColumnDimensions()[0], self.sensorLayer_sp_activeColumns.dense, subplot=122, drawPlot=True)
		else:
			self.UpdateHtmVisValues(sensorValue=sensedFeature,
									sensorSDR=sensorSDR,
									activeCellsSDR=activeCellsSDR,
									predictiveCellsSDR=predictiveCellsSDR,
									spatial_pooler_object=self.sensorLayer_sp,
									temporal_pooler_object=self.tm)

		#self.print_htm_state_info(enc_info, sp_info, tm_info, sp, tm)


	def print_htm_state_info(self, enc_info, sp_info, tm_info, sp, tm):
		# Print information & statistics about the state of the HTM.
	    print("Encoded Input", enc_info)
	    print("")
	    print("Spatial Pooler Mini-Columns", sp_info)
	    print(str(sp))
	    print("")
	    print("Temporal Memory Cells", tm_info)
	    print(str(tm))
	    print("")

	def plotBinaryMap(self, name, size, data, subplot=0, drawPlot=False):
		plotW = math.ceil(math.sqrt(size))

		rf = np.zeros([ plotW, plotW ], dtype=np.uint8)
		for i in range(plotW):
			arr = data[i*plotW:i*plotW+plotW]*2
			if len(arr)<plotW:
				arr=np.concatenate([arr, np.ones(plotW-len(arr))])
			rf[:, i] = arr

		if subplot>0:
			plt.subplot(subplot)

		plt.imshow(rf, interpolation='nearest')
		plt.title( name)
		plt.ylabel("Rows")
		plt.xlabel("Columns")

		if subplot>0:
			plt.tight_layout()

		if subplot==0 or subplot>0 and drawPlot:#if we are doing multiplot, draw only at the last call
			plt.show()


	def start_loop(self):

		print("Iteration:"+str(0))
		self.SystemCalculate()

		for i in range(5):
			print("Iteration:"+str(i+1))
			self.SystemCalculate()
			self.agent.moveDir(Direction.RIGHT)
			#time.sleep(1)



def animated_main():
	nx = 150
	ny = 50

	fig = plt.figure()
	data = np.zeros((nx, ny))
	im = plt.imshow(data, cmap='gist_gray_r', vmin=0, vmax=1)

	def init():
	    im.set_data(np.zeros((nx, ny)))

	def animate(i):
	    xi = i // ny
	    yi = i % ny
	    data[xi, yi] = 1
	    im.set_data(data)
	    return im

	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=nx * ny,
	                               interval=50)

def old_main():

	# load model parameters from file
	f = open('modelParams.cfg','r').read()
	modelParams = eval(f)

	# set up system
	SystemSetup(modelParams)

	# put agent in the environment
	agent.set_env(env,3,4)

	print("Iteration:"+str(0))
	SystemCalculate()

	for i in range(5):
		print("Iteration:"+str(i+1))
		SystemCalculate()
		agent.moveDir(Direction.RIGHT)
		time.sleep(1)


if __name__ == "__main__":
	object_2d = AnimateSDR2D()
	object_2d.start_loop()

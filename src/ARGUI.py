import sys
import cv2
import tkinter as tk
from PIL import ImageTk
from PIL import Image
import AR
import numpy as np

class ARGUI(tk.Frame):

	"""
	Frame that fills the window Tk().
	"""

	def __init__(self, picklePathScene, picklePathObj, 
			     scenePathLeft, scenePathRight, objectPath, segments=[]):

		self.AR = AR.AR(picklePathScene, picklePathObj,scenePathLeft, 
					    scenePathRight, segments, objectPath)

		#The Object image parameters
		self.x = -500
		self.y = -500
		self.theta = 0
		self.scaleX = 0.25
		self.scaleY = 0.25
		self.depth = 15

	def startGUI(self):

		"""
		Runs the GUI
		"""

		self.root = tk.Tk()
		tk.Frame.__init__(self, self.root)
		self.grid()

		[sceneImage, objectImage] = self.getImagesFromAR()
		
		self.sceneFrame = SceneFrame(self, sceneImage)
		self.objectFrame = ObjectFrame(self, objectImage, sceneImage.width())
		self.menuFrame = MenuFrame(self)

		self.objectFrame.grid(row=0)
		self.sceneFrame.grid(row=1)
		self.menuFrame.grid(row=2)

		self.updateImagesFromAR()
		self.root.mainloop()

	def updateImagesFromAR(self):

		"""
		Calls the AR class to get images updated with current parameters
		and displays these in the GUI
		"""

		[self.sceneFrame.img,self.objectFrame.img] = self.getImagesFromAR()
		self.objectFrame.canvas.itemconfig(self.objectFrame.canvasImage,
										   image = self.objectFrame.img)
		self.objectFrame.centerImage()

		self.sceneFrame.canvas.itemconfig(self.sceneFrame.canvasImage, 
										   image = self.sceneFrame.img)


	def getImagesFromAR(self):

		"""
		Gets the images from AR class and returns them in PhotoImage format
		"""

		[sceneImage, objectImage] = self.AR.getImages(x=self.x, y=self.y, 
											  theta=self.theta, scaleX=self.scaleX,
											  scaleY=self.scaleY, 
											  depthOffset=self.depth*1000)

		sceneImage = self.convertCVToPhoto(sceneImage)
		objectImage = self.convertCVToPhoto(objectImage)
		return [sceneImage, objectImage]


	def convertCVToPhoto(self, imgCV):

		"""
		Converts to PhotoImage format from CV

		Args:
			imgCV([[Double]])	: The opencv image
		"""

		#Change format to RGB
		b,g,r = cv2.split(imgCV)
		imgRGB = cv2.merge((r,g,b))

		#Convert to PhotoImage, storing in self to avoid garbage collector
		imgRGB = Image.fromarray(imgRGB)
		return ImageTk.PhotoImage(image = imgRGB)

	def exitGUI(self):
		self.objectFrame.grid_forget()
		self.sceneFrame.grid_forget()
		self.menuFrame.grid_forget()

		self.objectFrame.destroy()
		self.sceneFrame.destroy()
		self.menuFrame.destroy()

		self.grid_forget()
		self.destroy()

		self.root.destroy()

class ObjectFrame(tk.Frame):

	"""
	Frame containing the object to be put into the scene

	Args:
		parent(tk.Frame)	: The parent of the object
		image(PhotoImage)	: The object image
		maxWidth=1400(Int)	: The maximum width of the canvas
		maxHeight=200(Int)	: The maximum height of the canvas 
	"""

	def __init__(self, parent, img, width=1400, height=300):
		self.parent = parent
		tk.Frame.__init__(self, parent)
	
		self.canvas = tk.Canvas(self)

		self.canvas['height'] = height
		self.canvas['width'] = width
		self.img = img

		self.canvasImage = self.canvas.create_image(0,0, image=img, anchor="nw")
		self.canvas.grid()

	def centerImage(self):

		"""
		Moves the object image to the center of the canvas
		"""

		imgWidth = self.img.width()
		imgHeight = self.img.height()

		imgOffset = (imgWidth/2, imgHeight/2)
		canvasMid = (int(self.canvas['width'])/2, int(self.canvas['height'])/2)

		newX  = int(canvasMid[0] - imgOffset[0]) 
		newY = int(canvasMid[1] - imgOffset[1])

		self.moveImageTo(newX, newY)

	def moveImageTo(self, x, y):
		
		"""
		Moves the object image to the given coordinates

		Args:
			x(int) : The new x coordinate
			y(int) : The new y coordinate
		"""

		currentX,currentY = self.canvas.coords(self.canvasImage)
		offsetX, offsetY = x - currentX, y - currentY
		self.canvas.move(self.canvasImage, offsetX, offsetY)

class SceneFrame(tk.Frame):

	"""
	Frame containing the AR image

	Args:
		parent(tk.Frame)	: The parent of the object
		image(PhotoImage)	: The object image
		maxWidth=1400(Int)	: The maximum width of the canvas
		maxHeight=200(Int)	: The maximum height of the canvas 
	"""

	def __init__(self, parent, img, maxWidth=1400, maxHeight=400):
		self.parent = parent
		tk.Frame.__init__(self, parent)

		self.img = img
		self.canvas = tk.Canvas(self)
		
		if(maxHeight > 0 and img.height() > maxHeight):
			self.canvas['height'] = maxHeight
		else:
			self.canvas['height'] = img.height()

		if(maxWidth > 0 and img.width() > maxWidth):
			self.canvas['width'] = maxWidth
		else:
			self.canvas['width'] = img.width()
		
		self.img = img
		self.canvasImage = self.canvas.create_image(0,0, image=img, anchor="nw")
		self.canvas.bind("<Button-1>", self.leftclick)
		self.canvas.grid()

	def leftclick(self, event):

		"""
		Updates self.x and self.y if the scene image is clicked. 

		Args:
			event	: The click event
		"""

		x = int(self.canvas.canvasx(event.x)-1)
		y = int(self.canvas.canvasy(event.y)-1)
		if(x > 0 and x < self.img.width() and y > 0 and y < self.img.height()):
			self.parent.x = y
			self.parent.y = x
			self.parent.updateImagesFromAR()

class MenuFrame(tk.Frame):
	
	"""
	Frame containing the menu Buttons and Entries

	Args:
		parent(tk.Frame)	: The parent of the object
	"""
	
	def __init__(self, parent):

		self.parent = parent
		tk.Frame.__init__(self, parent)
		
		#Create theta entry widget
		tk.Label(self, text="Theta (Degrees):").grid(row=0, column=0)
		self.thetaEntry = tk.Entry(self, width = 5)
		self.thetaEntry.insert(0,str(self.parent.theta))
		self.thetaEntry.bind('<Return>', lambda event: self.update())
		self.thetaEntry.grid(row=0,column=1)

		#Create scaleX entry widget
		tk.Label(self, text="ScaleX:").grid(row=0, column=2)
		self.scaleXEntry = tk.Entry(self, width = 5)
		self.scaleXEntry.insert(0,str(self.parent.scaleX))
		self.scaleXEntry.bind('<Return>', lambda event: self.update())
		self.scaleXEntry.grid(row=0,column=3)

		#Create scaleY entry widget
		tk.Label(self, text="ScaleY:").grid(row=0, column=4)
		self.scaleYEntry = tk.Entry(self, width = 5)
		self.scaleYEntry.insert(0,str(self.parent.scaleY))
		self.scaleYEntry.bind('<Return>', lambda event: self.update())
		self.scaleYEntry.grid(row=0,column=5)

		#Create theta entry widget
		tk.Label(self, text="Depth (Meters):").grid(row=0, column=6)
		self.depthEntry = tk.Entry(self, width = 5)
		self.depthEntry.insert(0,str(self.parent.depth))
		self.depthEntry.bind('<Return>', lambda event: self.update())
		self.depthEntry.grid(row=0,column=7)

		tk.Button(self, command=lambda:self.update(), text="Update").grid(row = 0, column=8)
		tk.Button(self, command=lambda:self.save(), text="Save object").grid(row = 0, column=9)
		tk.Button(self, command=lambda:self.parent.exitGUI(), text="Exit").grid(row=0,column=10)

	def update(self):

		"""
		Updates the theta, scaleX and scaleY to current widget values and 
		reloades images
		"""

		if(not self.updateParameters()):
			return

		self.parent.updateImagesFromAR()

	def save(self):

		"""
		Saves the current object in the scene image
		"""

		if(not self.updateParameters()):
			return
		self.parent.AR.saveCurrentImage(x=self.parent.x, y=self.parent.y, 
										theta=self.parent.theta, 
										scaleX=self.parent.scaleX,
										scaleY=self.parent.scaleY, 
										depthOffset=self.parent.depth*1000)
	
	def updateParameters(self):

		"""
		Updates the parameters from the GUI Entries
		"""

		try:
			self.parent.theta = float(self.thetaEntry.get())
		except:
			print("Could not update theta value, enter a valid float number")
			return False
		try:
			self.parent.scaleX = float(self.scaleXEntry.get())
		except:
			print("Could not update scaleX value, enter a valid float number")
			return False
		try:
			self.parent.scaleY = float(self.scaleYEntry.get())
		except:
			print("Could not update scaleY value, enter a valid float number")
			return False
		try:
			self.parent.scaleY = float(self.scaleYEntry.get())
		except:
			print("Could not update scaleY value, enter a valid float number")
			return False
		try:
			self.parent.depth = float(self.depthEntry.get())
		except:
			print("Could not update depth value, enter a valid float number")
			return False

		return True

if __name__ == '__main__':
	sess = ARGUI("../Pickle/City1.p", "../Pickle/Porsche.p",
			     "../Images/CityLeft1.png", "../Images/CityRight1.png", 
				"../Images/Porsche.jpg", segments=[2,3,4,5,6,7,8,9,10,11,12])
	sess.startGUI()








''' IMPORT CHECK '''
# import cv2
# import numpy
# import sklearn
# import scipy
# import skimage
# import fastapi
# print("All libraries imported successfully!")

''' Image Load + Color Conv (RGB to Gray) Test '''
# from utils.image_io import loadImage, getGray, getResizedImage
# img = loadImage("data/screen/test.png")
# print(img.shape)
# gray = getGray(img)
# print(gray.shape)
# img = getResizedImage("data/screen/test.png")
# gray = getGray(img)
# saveImage(gray, "outputs/gray.png")

''' Test2'''
# from utils.image_io import *
# img = loadImage("data/screen/test.png")
# print("Loaded:", img.shape)
# img = resizeImage(img)
# print("Resized:", img.shape)
# gray = getGray(img)
# print("Gray:", gray.shape)
# rgb = getRGB(img)
# print("RGB:", rgb.shape)
# saveImage(gray, "outputs/gray.jpg")
# print("Done")

''' Testing Preprocessing Functions '''
# from utils.image_io import *
# from utils.preprocessing import *
# img = getResizedImage("data/screen/test.png")
# gray = getGray(img)
# eq = equalizeHistogram(gray)
# blur = gaussianBlur(gray)
# patches = splitPatches(gray)
# print("Patches:", len(patches))
# print("Entropy:", imageEntropy(gray))
# print("Contrast:", imageContrast(gray))
# saveImage(eq, "outputs/equalized.png")
# saveImage(blur, "outputs/blur.png")

'''Testing frequency'''
# from utils.image_io import *
# from evidence.frequency import *
# img = getResizedImage("data/screen/test.png")
# gray = getGray(img)
# result = freqE(gray)
# print(result)

'''testing geometry'''
# from utils.image_io import *
# from evidence.geometry import geometryEvidence
# from evidence.focus import focusEvidence
# img = getResizedImage("data/screen/test.png")
# gray = getGray(img)
# print("\nGeometry")
# print(geometryEvidence(gray))
# print("\nFocus")
# print(focusEvidence(gray))

'''testing compression'''
# from evidence.compression import compressionEvidence
# from utils.image_io import *
# img = getResizedImage("data/screen/test.png")
# gray = getGray(img)
# print("\nCompression")
# print(compressionEvidence(gray))

'''testing fusion/fts'''
# from fusion.features import extractFeatures
# from evidence.compression import compressionEvidence
# from utils.image_io import *
# img = getResizedImage("data/screen/test.png")
# gray = getGray(img)
# print("\nFeature Vector")
# features = extractFeatures(img)
# print(features)
# print(features.shape)
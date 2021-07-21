from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import pylab as plt
import numpy as np

def varianceCalculate(average, histgram):
    
    variance = 0
    for i in range(len(histgram)):
        variance += (histgram[i] - average) ** 2

    variance /= len(histgram)

    return variance

def averageAndpixelSumCalculate(histgram):
    
    average = pixelSum = 0
    for i in range(len(histgram)):
        pixelSum += histgram[i]             
        brightnessValue = histgram[i] * i   

    average = brightnessValue / len(histgram)

    return pixelSum, average

def within_betweenCV(pixelSum1, average1, variance1, pixelSum2, average2, variance2):
    

    betweenClassVariance = (pixelSum1 * pixelSum2 * ((average1 - average2) ** 2) ) / ((pixelSum1 + pixelSum2) ** 2)

    withinClassVariance = (pixelSum1 * variance1 + pixelSum2 * variance2) / (pixelSum1 + pixelSum2)

    return betweenClassVariance, withinClassVariance

def calculateAll(blackList, whiteList):
    
    b_size, b_average = averageAndpixelSumCalculate(blackList)
    w_size, w_average = averageAndpixelSumCalculate(whiteList)

    b_variance = varianceCalculate(b_average, blackList)
    w_variance = varianceCalculate(w_average, whiteList)

    betweenCV, withinCV = within_betweenCV(b_size, b_average, b_variance, w_size, w_average, w_variance)

    totalVariance = betweenCV + withinCV
    separationMetrics = betweenCV / (totalVariance - betweenCV)

    return separationMetrics

def main():
    image_path = "./m_r.jpg"                            
    image = cv2.imread(image_path, 0)                      

    
    histgram = cv2.calcHist([image], [0], None, [256], [0, 256])

    
    size = 256
    listSM = [0 for i in range(size)]                       
    for i in range(size):
        if i != 0 and i != size-1:
            blackList = histgram[0: i]                      
            whiteList = histgram[i: size]                   
            listSM[i] = calculateAll(blackList, whiteList) 
        elif i == 0 or i == size-1:
            listSM[i] = 0                                   

    
    maxValue = 0                                           
    for i in range(size):
        
        if listSM[i] > maxValue:
            maxValue = listSM[i]
            maxValueIndex = i                               

    output_otsu = np.zeros((len(image), len(image[0])))    
    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] > maxValueIndex:
                output_otsu[i][j] = 255                    
            else:
                output_otsu[i][j] = 0                      


    average_histgram = int(len(histgram) / 2)               
    output_average = image.copy()                           
    output_average[output_average >= average_histgram] = 255                  
    output_average[output_average < average_histgram] = 0                     


    aGH = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 89, 7)


    # cv2.imwrite("gray.jpg", image)                          
    # cv2.imwrite("average.jpg", output_average)              
    cv2.imwrite("./Segmented Images/otsu.jpg", output_otsu)                    
    # cv2.imwrite("Adaptive_Gaussian_Thresholoding.jpg", aGH) 

    cv2.imshow("input", image)                              
    cv2.imshow("average", output_average)                   
    cv2.imshow("otsu", output_otsu)                         
    cv2.imshow("adaptive gaussian", aGH)                    

    plt.plot(histgram)                                      
    plt.axvline(x=maxValueIndex, color='red', label='otsu') 
    plt.axvline(x=average_histgram, color='green', label='average')
    plt.legend(loc='upper right')                           
    plt.title("histgram of brightness")                     
    plt.xlabel("brightness")                                
    plt.ylabel("frequency")                                 
    plt.xlim([0, 256])                                      
    plt.savefig('./Segmented Images/histogram_sample.png')

if __name__ == '__main__':
    main()

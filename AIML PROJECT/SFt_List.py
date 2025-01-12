

def getSensors():
    sensorList = [ 
            'Accelerometer: x-axis (g)','Accelerometer: y-axis (g)',
            'Accelerometer: z-axis (g)','Gyroscope: x-axis (rad/s)',
            'Gyroscope: y-axis (rad/s)','Gyroscope: z-axis (rad/s)']  


    return sensorList

def getFeatures():
    ftList = [
        'Mean',
        'StandardDeviation',
        'RootMeanSquare',
        'MaximalAmplitude',
        'MinimalAmplitude',
        'Median',
        'Number of zero-crossing',
        'Skewness',
        'Kurtosis',
        'First Quartile',
        'Third Quartile',
        'Autocorrelation',
        'Energy'
        ] 
    return ftList
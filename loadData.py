
csvFile = open(r'data.csv','w')
csvFile.writelines('predictor_variable'+','+'target_variable')

for i in range(1,100000):
    csvFile.writelines('\n'+str(i)+','+str(i**2))

csvFile.close()
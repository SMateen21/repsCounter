import csv

"""
Called when capture button is pressed, saves to CSV
"""
def writeLandmarks(landmarks):
    file = csv.writer(open("landmarks.csv", 'a'), delimiter=',')
    file.writerow(landmarks)
    return


fieldname = ['status']
for i in range(21):
        x = 'x' + str(i)
        y = 'y' + str(i)
        z = 'z' + str(i)
        fieldname.append(x)
        fieldname.append(y)
        fieldname.append(z)
f = csv.writer(open("landmarks.csv", "w"), delimiter=',')
f.writerow(fieldname)

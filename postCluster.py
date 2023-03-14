import os
import csv
import shutil
import csv
import glob
import sys
import numpy as np


from imagededup.methods import PHash

#test path dir
rootPath = "/mnt/dl-storage/dg-cephfs-0/public/Ambilight/PBVS/track2_test_march_1/EO_test"
rootDir = "./cluster"
saveDir = ""
secondfile =  "secondResult1/class4_best_v0.csv"
firstfile = "firstResult1/*.csv"
resulttemp = "results_temp.csv"
resultfile = "results.csv"

# imgid, label, score
class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def imgid(self):
        return int(self._data[0])

    @property
    def classid(self):
        return int(self._data[1])

    @property
    def score(self):
        return float(self._data[2])



def hashEncoder(rootDir):
    
    phasher = PHash()

    # hashencoder
    encodings = phasher.encode_images(image_dir = rootDir)

    # find same image
    d_1 = phasher.find_duplicates(encoding_map=encodings)

    # show same image
    """
    from imagededup.utils import plot_duplicates
    plot_duplicates(image_dir='cluster',
                    duplicate_map=d_1,
                    filename='cluster/Gotcha17019560.png')

    """
    return d_1



def secondClassifty(secondfile):
    
    dic_label_4 = {}
    dic_score_4 = {}

    filereader = open(secondfile, 'r')
    csvreader = csv.reader(filereader)
    col = 0
    for line in csvreader:
        col +=1
        if col ==1:
            continue
        imgid, classid, score = line
        dic_label_4[imgid] = int(classid)
        dic_score_4[imgid] = float(score)
    return dic_label_4, dic_score_4
    

def clusterShow(rootDir, saveDir):
    d_1 = hashEncoder(rootDir)
    for k, v in d_1.items():
        if len(v) > 20:
            if not os.path.exists(saveDir, k):
                os.makedirs(k)
        for v_ in v:
            shutil.copy(os.path.join(rootDir,v_), os.path.join(saveDir, v_))


def cluster(secondfile):
    # f = open("id.txt","w")
    clusterNum = 20
    thread2 = 0.95

    dicNew = {}
    if len(os.listdir(rootDir)) < 1:
        print ("rootDir is error! please check!")
    d_1 = hashEncoder(rootDir)
    dic_label_4, dic_score_4 = secondClassifty(secondfile)

    for k, v in d_1.items():
        imgid = k[6:-4]
        curlabel = dic_label_4[imgid]

        tempNew = {}
        if len(v) > clusterNum:
            templabels = []
            tempscores = 0
            temp = 0
            diffnum = 0
            tempNew[imgid] = curlabel
            
            for v_ in v:
                vid = v_[6:-4]
                tempscore = dic_score_4[vid]
                tempscores += tempscore
                templabel = dic_label_4[vid]
                templabels.append(templabel)
                temp +=1
                print (vid, templabel)
                
                tempNew[vid] = templabel
                if vid in dicNew and dicNew[vid]!=templabel: 
                    diffnum +=1
            

            if len(set(templabels)) == 1 and len(templabels) == len(v):
                if diffnum == 0 and templabels[0] == curlabel:
                    if tempscores / temp > thread2:
                        dicNew.update(tempNew)

    # reset label
    #print (len(dicNew))
    #for key in dicNew:
    #    f.write(key + "\n")
    return dicNew
  

def firstClassifty(firstfile, resulttemp):
    
    files = glob.glob(firstfile)
    numsfile = len(files)
    resultDic = {}
    lis = []
    thresh = 0.9

    for i, file in enumerate(files):
        path = file.split("/")[-1]
        lis.append(path)

        if path not in resultDic.keys():
            resultDic[path] = {}

            filereader = open(file, 'r')
            csvreader = csv.reader(filereader)
            col = 0
            for line in csvreader:
                col +=1
                if col ==1:
                    continue
                dataInfo = VideoRecord(line)  # imgid, classid, score
                imgid, classid, score = line
                
                if imgid not in resultDic[path].keys():
                    resultDic[path][imgid] = dataInfo
                else:
                    print ("data error, please check!", file, imgid)
    
    voteres = {}
    resulttemp = open(resulttemp, 'w', newline='')
    writer = csv.writer(resulttemp)
    writer.writerow(['image_id', 'class_id', 'score'])
    jdx = 0
    for id_, dataInfo in  resultDic[lis[0]].items():
        curlis = []
        for path in lis:
            print ("path")
            #import pdb;pdb.set_trace()
            # imgid, classid, score = resultDic[path][id_]
            curlis.append(resultDic[path][id_]._data[1])

        maxlabel = max(curlis,key=curlis.count)
        maxnum = curlis.count(maxlabel)
        score = 0
        num = 0
       
        if maxnum >= numsfile:
            for path in lis:
                #print ("label:", resultDic[path][id_]._data[1])
                #print ("score:", resultDic[path][id_]._data[2])
                if resultDic[path][id_]._data[1] == maxlabel:
                    score += float(resultDic[path][id_]._data[2])
                    num += 1
            cur_label = maxlabel
            cur_score = score / num
            if cur_label == "0":
                if cur_score > thresh:
                    curPath = os.path.join(rootPath, "Gotcha" + str(id_) + ".png")
                    newPath = os.path.join(rootDir, "Gotcha" + str(id_) + ".png")
                    shutil.copy(curPath, newPath)
                    writer.writerow([int(id_), int(cur_label), float(cur_score)])
            else:
                writer.writerow([int(id_), int(cur_label), float(cur_score)])


def updateResult(resulttemp, resultfile):
    resfile = open(resultfile, 'w', newline='')
    writer = csv.writer(resfile)
    dicNew = cluster(secondfile)
    lis = []
    for id in dicNew.keys():
        lis.append(id)
        
    filereader = open(resulttemp, 'r')
    csvreader = csv.reader(filereader)
    jdx = 0
    i = 0
    tempresult = []
    for line in csvreader:
        i+=1
        if i == 1:
            writer.writerow(line)
        else:
            imgid, classid, score = line
            if classid!="0":
                tempresult.append([int(imgid), int(classid), float(score)])
                writer.writerow([int(imgid), int(classid), float(score)])
                jdx +=1
            else:
                if imgid in lis:
                    tempresult.append([int(imgid), int(classid), float(score)])
                    writer.writerow([int(imgid), int(classid), float(score)])
                    jdx +=1
    

   
    

def main():
    if not os.path.exists(rootDir):
        os.makedirs(rootDir)
    firstClassifty(firstfile, resulttemp)
    updateResult(resulttemp, resultfile)
    print ("infer Done! result file path is:", resultfile)
     
      
if __name__ == "__main__":
    main()

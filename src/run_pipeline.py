import os, sys
from pipeline_scripts.mask_context_evaluation import Context_Iterator
from pipeline_scripts.contour_extraction import Contour_Iterator
from pipeline_scripts.metrics import Metrics_Iterator
import pipeline_scripts.utils
import time

def getTrialFrames(cwd,task,trial):
    TrialRoot = os.path.join(cwd,task,"images",trial)
    for root, dirs, files in os.walk(TrialRoot):
        Frames = files
        break    
    
    frameNumbers = []
    for f in Frames:
        frameNumber = f.replace(".png","").split("_")[1]
        frameNumbers.append(frameNumber)
    #print("Frames",Frames,"\nFNums",frameNumbers)
    #print("FNums",frameNumbers)
    return frameNumbers

def main():
    CWD=os.getcwd()
    TASK = "Needle_Passing" #DEFAULT 
    try:
        TASK=sys.argv[1]
    except:
        print("Error: no task provided","Usage: python context_gen_pipeline.py <task>","Default Task"+TASK)
    taskDir = os.path.join(CWD, TASK,"images")
    trials = [] 
    for root, dirs, files in os.walk(taskDir):
        trials = dirs
        break
        
    
    BATCH_SIZE = 25
    EPOCH = 0
    TRIAL = trials[0]
    TRIAL_FRAMES = getTrialFrames(CWD,TASK,TRIAL) # formatted as the XXXX in 'frame_XXXX.png'
    TOTAL_TIME = 0
    TOTAL_EPOCH = 0
    

    print("Pipeline Running for task ",TASK,"trial",TRIAL)
    while(EPOCH * BATCH_SIZE < len(TRIAL_FRAMES)):
        print("EPOCH ", EPOCH, ":", TRIAL_FRAMES[BATCH_SIZE*EPOCH:BATCH_SIZE*EPOCH+BATCH_SIZE])
        start_time = time.time()
        I = Contour_Iterator(TASK,CWD)
        # Format: ["2023_grasper_L_masks","other folders"], ["2023_grasper_L", "label names"], ["C://...//2023_grasper_L_masks/Knot_Tying_S03_T02_0.json", "other filenames"] 
        label_classes, label_classNames, ContourFiles, RingFile = I.ExtractContours(BATCH_SIZE,EPOCH,TRIAL,TRIAL_FRAMES[BATCH_SIZE*EPOCH:BATCH_SIZE*EPOCH+BATCH_SIZE])
        I = Context_Iterator(TASK,CWD)
        I.GenerateContext(BATCH_SIZE,EPOCH,TRIAL,TRIAL_FRAMES[BATCH_SIZE*EPOCH:BATCH_SIZE*EPOCH+BATCH_SIZE],label_classes, label_classNames,ContourFiles,RingFile,SAVE=True)
        end_time = time.time()
        epoch_time = end_time - start_time
        if((EPOCH+1) * BATCH_SIZE < len(TRIAL_FRAMES)):
            TOTAL_TIME += epoch_time
            TOTAL_EPOCH +=1
        print("   %s s \n" % round(epoch_time,3), end="")
        EPOCH +=1

    print("\n---------------------- Time Analysis --------------------")
    print("Average runtime for batch size=",BATCH_SIZE," is ", round(TOTAL_TIME/TOTAL_EPOCH,4),"seconds")
    print("                                        ", round(TOTAL_TIME/TOTAL_EPOCH,6)*1000,"ms")
    print("Total Execution time",round(TOTAL_TIME,5),"seconds")
    print("\n---------------------- Metrics --------------------")
    I = Metrics_Iterator(TASK,CWD)
    I.IOU()
    quit();

'''
def getLabelClassnames(task):
    if "Knot" in task:
        return ["2023_grasper_L_masks","2023_grasper_R_masks","2023_thread_masks"], ["2023_grasper_L","2023_grasper_R","2023_thread" ]
    elif "Needle" in task:
        return ["2023_grasper_L_masks","2023_grasper_R_masks","2023_thread_masks"], ["2023_grasper_L","2023_grasper_R","2023_thread" ] # add Needle,
    elif "Suturing" in task:
        return ["2023_grasper_L_masks","2023_grasper_R_masks","2023_thread_masks"], ["2023_grasper_L","2023_grasper_R","2023_thread" ] # add Needle
'''


main();

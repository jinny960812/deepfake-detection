import os

images = os.listdir(r"E:\Celeb-DF-v2\Celeb-real_images")
count = 0
i = 0
file_list = open("List_training_1.txt", "w")
file_list2 = open("List_testing_1.txt", "w")
f_list = file_list

images = os.listdir(r"E:\Celeb-DF-v2\Celeb-real_images")
for image in images:
    if not (i%10) :
        fname = os.path.join("E:\Celeb-DF-v2\Celeb-real_images", image)
        if (os.path.getsize(fname)!=0):
            f_list.write(fname + ' 1 \n')
            count+=1
        if count==10000:
            f_list=file_list2
        if count== 12500:
            break
    i+=1

count=0
f_list = file_list
images = os.listdir(r"E:\Celeb-DF-v2\Celeb-synthesis_images")
for image in images:
    if not (i%30) :
        fname = os.path.join("E:\Celeb-DF-v2\Celeb-synthesis_images", image)
        if (os.path.getsize(fname) != 0):
            f_list.write(fname + ' 0 \n')
            count+=1
        if count==10000:
            f_list = file_list2
        if count == 12500 :
            break
    i+=1

file_list.close()
file_list2.close()




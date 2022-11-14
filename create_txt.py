def create_txt(date,label,num):
    # date = ['2022-11-05', '2022-11-05', '2022-11-06', '2022-11-07']
    # label = ["a","b","c","d"]
    # num = [1,2,3,4]
    # print(len(date))
    # print(len(label))
    # print(len(num))


    def txt_create(name, msg):
        desktop_path = "fireflies/txt/"  # 新创建的txt文件的存放路径
        full_path = desktop_path + name + '.txt'  # 也可以创建一个.doc的word文档
        file = open(full_path, 'w')
        file.write(msg)  # msg也就是下面的Hello world!
        # file.close()

    if  len(date) == len(label) == len(num):
        for i in range(len(date)):
            txt_create(str(label[i])+"."+str(date[i]),str(date[i])+" "+str(num[i]))
    else:
        print("参数不对")

date = ['2022-11-05', '2022-11-05', '2022-11-06', '2022-11-07']
label = ["a","b","c","d"]
num = [1,2,3,4]

create_txt(date,label,num)







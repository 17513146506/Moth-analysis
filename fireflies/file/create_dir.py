def create_dir(date):
    import os
    import shutil


    path = "fireflies/txt/"
    def make_dir(date):
        date = set(date)#去除多余的部分
        #path = "fireflies/txt/"
        for dat in date:
            dir_path = os.path.join(path,dat)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

    def move_path(path):
        dirs = os.listdir(path)
        print(dirs)
        for dir in dirs:
            if os.path.isfile(os.path.join(path,dir)):
                print("aa")
                dat = dir.split(".",-1)[1]
                dir_path = os.path.join(path,dir)
                new_path = os.path.join(path,dat)
                new_name = dir.split(".",-1)[0] + ".txt"
                new_name = "/" + new_name
                shutil.move(dir_path,new_path+new_name)


    make_dir(date)
    move_path(path)








'''
The file path that related to all project are listed as follow.
'''

MAX_COMMENT_COUNT = 200
MAX_CONTENT_LENGTH =512
import argparse 
import datetime
parse = argparse.ArgumentParser(description="ReFEND param")
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print("Current time:", formatted_time)
time = formatted_time
parse.add_argument("--time",type=str,default=time)


############################################################Weibo-comp##############################################33

parse.add_argument('--num_epochs',type=int,default=100)
parse.add_argument("--Batch_size",type=int,default=64)
parse.add_argument("--hidden_dim",type=int,default=512)
parse.add_argument("--learning_rate",type=float,default=1e-3)
parse.add_argument("--weight_decay",type=float,default=0)
parse.add_argument("--class_num",type=int,default=2)
parse.add_argument("--dataset",type=str,default="Weibo-comp")

args =parse.parse_args() 


######################################################Weibo20###########################################################

# parse.add_argument('--num_epochs',type=int,default=100)
# parse.add_argument("--Batch_size",type=int,default=64)
# parse.add_argument("--hidden_dim",type=int,default=256)
# parse.add_argument("--learning_rate",type=float,default=1e-4)
# parse.add_argument("--weight_decay",type=float,default=0)
# parse.add_argument("--class_num",type=int,default=2)
# parse.add_argument("--dataset",type=str,default="Weibo20")


# args =parse.parse_args() 


######################################################PolitiFact###########################################################

# parse.add_argument('--num_epochs',type=int,default=100)
# parse.add_argument("--Batch_size",type=int,default=16)
# parse.add_argument("--hidden_dim",type=int,default=256)
# parse.add_argument("--learning_rate",type=float,default=5e-4)
# parse.add_argument("--weight_decay",type=float,default=0)
# parse.add_argument("--class_num",type=int,default=2)
# parse.add_argument("--dataset",type=str,default="PolitiFact")


# args =parse.parse_args() 

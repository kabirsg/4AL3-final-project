# 4AL3-final-project
Final Project for SFWRENG 4AL3



# Running on Lambda Cloud

On web browser go to:
     cloud.lambdalabs.com

Login with:
ctaylor11235@gmail.com

Under SSH Keys, add your ssh key file (~/.ssh/id_rsa.pub) 

Under storage, there is a file system called "eye".
Note, the region is 'Washington DC, USA'
Also not the 'size' used is listed.  Lets try to keep this under 10G (i.e. remove old models).

Under Instances, it will list your current instances.
NOTE: if listed here they are charging a per-hour cost.  When not used, Delete. 

Fire up a new instance to use by 
  - clicking 'Launch Instance'
  - select "1x GH200 (96GB)"
  - select 'Washington DC, USA' (the only option)
  - select file system 'eye'
  - select your ssh key
  - click 'Launch Instance'
  - click 'I agree to above'

It should now show your instance in the 'Instances' page.  
Initially it will say 'Booting' under Status. 

After a minute or two the status will be 'Running', 
and the SSH Login will show the login command (ssh ubuntu@x.x.x.x), click this to copy to the clipboard.


In Mac terminal:

Paste that command and hit Enter.  
Type 'yes' and hit enter to confirm new fingerprint.

Note: You now are talking to your rented server computer, just like you are in a terminal window on that computer.
What you run here is running on the rented computer.

cd into the 'eye' directory which is your persistant filesystem.
   cd eye
Note: file in the home directory (/home/ubuntu) will be lost when you delete down this server, but files in the 'eye' directory will remain as they are in the seperate file system

You can 'ls' and 'cat' files.  
You can run your python code (no need to a venv, what is installed by default works)

You don't need this now because they already exist, but if for some reason you need to create a new file system:
   mkdir eyedata
   mkdir logs
   mkdir models


Open a new Mac terminal in your x4AL3-final-project directory:

Note: from here you will copy files to and from the server.

get the username IP address into a variable (do this again if you start a new instance)
   export L=ubuntu@X.X.X.X
with X.X.X.X replaced with your instance IP address

Rsync is a command that copies files from one computer to another using the ssh keys.  It can update a bunch of files at once but is cleverly only copies files that have changed.

This command will update the files in the your current directory (x4AL3-final-project) to the 'eye' directory on the server
   rsync -av --no-recursive * $L:/home/ubuntu/eye
Note: do this whenever you edit your code on your mac and want to be able to run the new code on the server.
Note: the '--no-recursive' is important, or else rsync will update all the data files, some of which we don't need.

You don't need to do this because I already did, but to get the data files onto the server
   rsync -av eyedata/output_images $L:/home/ubuntu/eye/eyedata/
   rsync -av eyedata/entire_dataset $L:/home/ubuntu/eye/eyedata/


Back in the terminal window that is ssh'ed into the Server now:

You can just run your training code, e.g.:
    python Fundus_Final_project_good.py restart eyedata/output_images
or
    python Fundus_Final_project_good.py restart eyedata/entire_dataset

watch the results.  You can kill it with ctrl-c.  

You can see the result models in the models directory
    ls models


Back in the Mac terminal:

You can pull back log files by rsync'ing the other way:
   mkdir remote
   rsync -av $L:/home/ubuntu/eye/logs remote
You will now have a copy of all the remote log file in a 'remote' directory

You can copy back a model file by using 'scp', a simple file copy that uses your ssh keys.
E.g.
   scp $L:/home/ubuntu/eye/models/model_file_20241214-141115_final_torch remote
(but replace the model file with what ever file you want - you can copy and paste from the other terminal where you 'ls'd the filenames)

Run 'python Test.py remote/model_file_20241214-141115_final_torch' (locally) to see the results.










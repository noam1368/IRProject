﻿INSTANCE_NAME="instance-3"
REGION=us-central1
ZONE=us-central1-c
PROJECT_NAME="ass3-370307"
IP_NAME="$PROJECT_NAME-ip"
GOOGLE_ACCOUNT_NAME="ofeklut" # without the @post.bgu.ac.il or @gmail.com part

# 0. Install Cloud SDK on your local machine or using Could Shell
# check that you have a proper active account listed
gcloud auth list 
# check that the right project and zone are active
gcloud config list
# if not set them
gcloud config set project $PROJECT_NAME
gcloud config set compute/zone $ZONE

# 1. Set up public IP
gcloud compute addresses create $IP_NAME --project=$PROJECT_NAME --region=$REGION
gcloud compute addresses list
# note the IP address printed above, that's your extrenal IP address.
# Enter it here: 
INSTANCE_IP="34.134.211.3"

# 2. Create Firewall rule to allow traffic to port 8080 on the instance
gcloud compute firewall-rules create default-allow-http-8080 \
  --allow tcp:8080 \
  --source-ranges 0.0.0.0/0 \
  --target-tags http-server

# 3. Create the instance. Change to a larger instance (larger than e2-micro) as needed.
gcloud compute instances create $INSTANCE_NAME \
  --zone=$ZONE \
  --machine-type=e2-standard-8\
  --network-interface=address=$INSTANCE_IP,network-tier=PREMIUM,subnet=default \
  --metadata-from-file startup-script=startup_script_gcp.sh \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --tags=http-server
# monitor instance creation log using this command. When done (4-5 minutes) terminate using Ctrl+C
#חשוב שהשינויים של זיכרון הדיסק של האינסטנס יעשו לפני הנקודה הזו אחרת הוא לא יכול להיכנס לאינסטנס
gcloud compute instances tail-serial-port-output $INSTANCE_NAME --zone $ZONE

# 4. Secure copy your app to the VM
#gcloud compute scp LOCAL_PATH_TO/search_frontend.py $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME
gcloud compute scp search_frontend.py $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME
gcloud compute scp inverted_index_gcp.py $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME



#stop machine before change type of machine
#gcloud compute instances stop $INSTANCE_NAME
#change machine 
#gcloud compute instances set-machine-type $INSTANCE_NAME --machine-type n1-standard-4
#start the machine
#gcloud compute instances start $INSTANCE_NAME



# Copy the files from your local machine to the instance
#gcloud compute scp --recurse posting_locations $INSTANCE_NAME:~/



# 5. SSH to your VM and start the app
gcloud compute ssh $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME

#resize disk
gcloud compute disks resize persistent-disk-0 --size 30
gcloud compute disks describe my-disk --zone=$ZONE

# Copy the files from the bucket to your local machine do it from inside the instance
gsutil cp -r gs://posting_locations/ .
#installs
pip install pyspark
pip install pandas
pip install nltk
#run
python3 search_frontend.py

################################################################################
# Clean up commands to undo the above set up and avoid unnecessary charges
gcloud compute instances delete -q $INSTANCE_NAME
# make sure there are no lingering instances
gcloud compute instances list
# delete firewall rule
gcloud compute firewall-rules delete -q default-allow-http-8080
# delete external addresses
gcloud compute addresses delete -q $IP_NAME --region $REGION






#####

rm search_frontend.py
exit
rm search_frontend.py
#now upload manually
gcloud compute scp search_frontend.py $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME:/home/$GOOGLE_ACCOUNT_NAME
gcloud compute ssh $GOOGLE_ACCOUNT_NAME@$INSTANCE_NAME
ls
python3 search_frontend.py





 gcloud compute ssh --zone $ZONE --project $PROJECT_NAME --tunnel-through-iap $INSTANCE_NAME
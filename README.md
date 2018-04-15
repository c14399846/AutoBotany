# AutoBotany


How to setup + install (rought overview):

* Ubuntu 16.04 is the compute server used *

* 'install_scripts/' has <i>most</i> of the instructions for installing stuff *

Need To setup a Gcloud, AWS, or equivalent cloud service setup

Compute Server / AWS EC2
1)  Need OpenCV installed (version 3+) on Python 3
2)  Need NodeJS installed (version 4.9+)
3)  Need Postgres installed (version 9/10)
4)  Install all python dependencies in your Anaconda work environment (or globally)
5)  Install all NodeJS packages (npm init in base folder of AutoBotany, thne npm install)
    (NB) Gcloud is quite a large install, only the storage stuff is really used for the Buckets. 
         Only import / install that if you want to save space

Postgres (SQL) / Amazon RDS
1)  Need Postgres installed (version 9/10) and to be hosted publicly

Cloud Functions / AWS Lambda
1)  Import 'bucketInput.zip' into the Gcloud cloud function suite
2)  Will make use of NodeJS (NB: Gcloud only has support for NodeJS, AWS has support for NodeJS, Java, and Python )


Get Access setup for IAM:
1)  Need read and write access to an input bucket
2)  Need write access to an output bucket
3)  Need access to the Postgres database


Change all the ip addresses used in nodeServer/filesave.js, and bucketInput Cloud function




How to run: 
1)  'cd nodeServer'
2)  'workon cv' or any other Anaconda / Pytohn work environment
3)  'node filesave.js'
4)  Then use the browser or Input Bucket to upload an image
    E.G 'http://your-ip-here:3000/'




# RUN in the base directory of AutoBotany

sudo apt-get update
sudo apt-get install nodejs
sudo apt-get install npm


curl -sL https://raw.githubusercontent.com/creationix/nvm/v0.33.8/install.sh -o install_nvm.sh

bash install_nvm.sh 
source ~/.profile
nano ~/.profile

workon cv
nvm ls-remote

nvm install 4.9.1
nvm use 4.9.1

npm -v


npm install express@4.16.3
npm install formidable@1.1.1
npm install google-cloud@0.57.0
npm install massive@4.7.1
npm install pg@7.4.1
npm install python-shell@0.5.0
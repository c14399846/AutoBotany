"use strict";

//https://www.w3schools.com/nodejs/nodejs_uploadfiles.asp
var http = require('http');
var formidable = require('formidable');
var fs = require('fs');
var ps = require('python-shell');

var scriptLoc = './imageProcess.py';
let imageExists = false;
var fpath = '';
var imgFile = null;

var gcloud = require('google-cloud');

var gcs = gcloud.storage({
  projectId: 'image-processing-189813',
  keyFilename: './gcs/keyfile.json'
});


// Reference an existing bucket.
var inBucketName = 'fypautobotany';
var outBucketName = 'fypautobotanyoutput'
var inputBucket = gcs.bucket(inBucketName);
var outputBucket = gcs.bucket(outBucketName);                


//const massive = require('massive');
const express = require('express');
const app = express();
const pg = require('pg');
const conString = "postgresql://postgres:imageskydatabase@35.205.117.19:5432/postgres";

const connectionInfo = {
	host : "35.205.117.19",
	port : 5432,
	database : "postgres",
	user: "postgres",
	password: "imageskydatabase"
};



function readPlantStream(pFile){
	return new Promise( function(resolve, reject) {

		let localReadStream = null;
		localReadStream = fs.createReadStream(pFile);
		return resolve(localReadStream);

	});
}


function writePlantStream(pFile){
	return new Promise( function(resolve, reject){

		let remoteWriteStream = null;
		remoteWriteStream = outputBucket.file(pFile).createWriteStream();
		return resolve(remoteWriteStream);

	});
}


function uploadPlantStream(localReadStream, remoteWriteStream){

	return new Promise( function(resolve, reject) {

		console.log("pipe start");
		let res = null;

		localReadStream.pipe(remoteWriteStream)
		  .on('error', function(err) {})
		  .on('finish', function() {
		  	
		  	console.log("pipe fin");
			res = 'Uploaded to bucket';
		    
		    return resolve(res);
		});

	});
}

function pyTest(){

	console.log("\nExists: " + imageExists + "\n");

	if(imageExists){

		var options = {
			mode: 'text',
			args: [fpath, imgFile]	
		};

		//console.log('fpath before python: ' + fpath);
		//console.log(options);

		ps.run(scriptLoc, options, function(err, results){

			if(err) throw err;
			console.log('Finished running py script');

			imageExists = false;


			var fileDir = results[1];
			var plantProcessedFilename = results[2];
			var plantContoursFilename = results[3];
			var plantID = results[4];
			var width = results[5];
			var height = results[6];

			
			if (!width && !height) {
				console.log("Null or Empty measurements");
				width = -1;
				height = -1;
			}

			var contFileLoc = fileDir + plantContoursFilename;
			var bucketImg = plantContoursFilename;


			outputBucket
				.upload(contFileLoc)
				.then( () => {
					console.log(`Uploaded output file ${contFileLoc} to ${outBucketName}`);
					pg.connect(conString, (err, client, done) => {
	
				    	if(err){
				    		console.log(err);
				    		done();
				    	}

				    	console.log("Connected to DB");
						
						
						//let username = "plantguy1";
				    	let day = -1;
						let date = new Date();

						
						let plantEvent = "none";
						let plantType = "pea";
						let growthCycle = "";
						let imgID = -1;
						
						let iBucket = inBucketName;
						let oBucket = outBucketName;
						
						let inputImgDir = "/";
						let outputImgDir = "/";
						
						let inputImg = imgFile;
						let outputImg = bucketImg;

						try {
							client.query(
					    		"INSERT INTO plants (plantID, plantType, growthCycle,\
					    		imgID, inputBucket, inputImg, inputImgDir, outputBucket, outputImg, outputImgDir, \
					    		day, date, plantEvent, \
					    		width, height) \
					    		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)",
					    	[plantID, plantType, growthCycle, 
					    	imgID, iBucket, inputImg, inputImgDir, oBucket, outputImg, outputImgDir, 
					    	day, date, plantEvent,
					    	width, height]);

					    	console.log("INSERT");

						} catch (error){
							console.log(error);
						}

						console.log("DB Inserted data");

						try{
				    		fs.unlink(fileDir + plantProcessedFilename);
				    		fs.unlink(fileDir + plantContoursFilename);
				    		fs.unlink(fileDir + imgFile);
						} catch(unlinkErr) {
							console.log(unlinkErr);
						}

				    }); // END POSTGRES CONNECT
				})
				.catch(err => {
					console.log("ERROR: ", err);
				}); 
		});

	}
}



// NEED TO FIX AND USE LATER ON
function readFile(oldPath, newPath){

	fs.rename(oldPath, newPath, function (err) {

		if (err) throw err;

		imageExists = true;
		//console.log("Exists in rename: " + imageExists);

		//fpath = oldPath;
		fpath = filePath;
		imgFile = files.filetoupload.name;
		pyTest();

		res.write('File uploaded!');
		res.end();

	});

}

//var upload_path = "/home/image/node/";

http.createServer(function (req, res) {

  if (req.url == '/fileupload') {

  	console.log("Server accessed from Bucket or CURL");

    var form = new formidable.IncomingForm();

    var tmpfile = '';


    form.parse(req, function (err, fields, files) {

    	// The bucket portion of code
	   	if(Object.keys(fields).length !== 0){

	   		console.log("Server accessed from bucket upload");

	   		var ifile = gcs.bucket(fields.bucket).file(fields.filename); 
	   		
	   		var local_path = './images/';

	   		var local_ifile = local_path + fields.filename;

	   		ifile.createReadStream()
	   			.pipe(fs.createWriteStream(local_ifile))
	   			.on('error', function(err) {})
	   			.on('response', function(response) {})
	   			.on('end', function(){})
	   			.on('finish', function(){
	   				console.log("have ifile\n");
	   				console.log(local_ifile);

	   				fpath = local_ifile;
	   				imgFile = fields.filename;

	   				pyTest();

					res.write('File uploaded!');
					res.end();
	   			});
	   	}

	   	// The GUI portion of the code (from browser)
	   	else {

			console.log("Server accessed from browser");
			//console.log(files);

			var oldPath = files.filetoupload.path;
			var newPath = './images/';
			var filePath = newPath + files.filetoupload.name;


			fs.rename(oldPath, filePath, function (err) {

				if (err) throw err;

				inputBucket
					.upload(filePath)
					.then( () => {
						console.log(`Uploaded input file ${filePath} to ${inBucketName}`);
					});

				// Issue found, shouldn't upload to buck and then run the pytohn code, 
				// as the inserted image is re-run because of the file uplaod to the bucket
				/*imageExists = true;
				console.log("Exists in rename: " + imageExists);

				//fpath = oldPath;
				fpath = filePath;
				imgFile = files.filetoupload.name;
				pyTest();
				*/
				res.write('File uploaded!');
				res.end();
	 			
			});

		}
    });

  imageExists = true;

  } else {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write('<form action="fileupload" method="post" enctype="multipart/form-data">');
    res.write('<input type="file" name="filetoupload"><br>');
    res.write('<input type="submit">');
    res.write('</form>');
    return res.end();
  }
}).listen(3000); 

"use strict";

//https://www.w3schools.com/nodejs/nodejs_uploadfiles.asp
var http = require('http');
var formidable = require('formidable');
var fs = require('fs');
var ps = require('python-shell');


//var scriptLoc = './../cs.py';
var scriptLoc = './imageProcess.py';
//var pyshell = new ps(scriptLoc);
let imageExists = false;
var fpath = '';
var imgFile = null;

var gcloud = require('google-cloud');

var gcs = gcloud.storage({
  projectId: 'image-processing-189813',
  keyFilename: './gcs/keyfile.json'
});


// Reference an existing bucket.
var inputBucket = gcs.bucket('fypautobotany');
var outputBucket = gcs.bucket('fypautobotanyoutput');                


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

		console.log('fpath before python: ' + fpath);
		console.log(options);

		//var pyshell = new ps(scriptLoc);
		//console.log(imageExists);
		ps.run(scriptLoc, options, function(err, results){

			if(err) throw err;
			console.log('Finished py script');

			imageExists = false;

			//console.log(results);

			console.log("Width:" + results[4]);
			console.log("Height:" + results[5]);

			// Hardcoded for testing
			//var localReadStream = fs.createReadStream('./images/pContours.png');

			var contFileLoc = results[1] + results[3];
			//console.log(contFileLoc);

			//var bucketImg2 = 'pContours' + '_' + imgFile;
			//var bucketImg = '\'' + results[3] + '\'';
			var bucketImg = results[3];
			//console.log(bucketImg);
			//console.log(bucketImg2);

			var localReadStream = null;
			var remoteWriteStream = null;
			//var localReadStream = fs.createReadStream(contFileLoc);
			//var remoteWriteStream = outputBucket.file(bucketImg).createWriteStream();

			var readPlant = readPlantStream(contFileLoc);

			readPlant.then( function(value){

				localReadStream = value;
				//console.log(localReadStream);
				//console.log("read");

				var writePlant = writePlantStream(bucketImg);
				writePlant.then( function(value){
					remoteWriteStream = value;
					//console.log(remoteWriteStream);
					//console.log("write");

					if(localReadStream != null && remoteWriteStream != null){

						console.log("ayy ");

						localReadStream.pipe(remoteWriteStream)
						  .on('error', function(err) {})
						  .on('finish', function() {
						  	console.log("LMAO");

						  	/*pg.connect(conString, (err, client, done) => {
			    	
						    	if(err){
						    		done();
						    		console.log(err);
						    	}
								
								let username = "plantguy1";
						    	let date = new Date();


						    	client.query(
						    		"INSERT INTO plants (plantID, plantType, growthCycle,\
						    		imgId, inputImg, inputImgDir, outputImg, outputImgDir, \
						    		day, date, event, \
						    		width, height) \
						    		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
						    	[plantID, plantType, growthCycle, 
						    	imgId, inputImg, inputImgDir, outputImg, outputImgDir, 
						    	day, date, event,
						    	width, height]);

						    }); // END POSTGRES CONNECT
						    */


						  });

					}

					/*uploadPlantStream(localReadStream, remoteWriteStream).then( function(value){
						console.log(value);
					});*/
					
					//console.log("pipe");
				});

			});

			/*var writePlant = writePlantStream(bucketImg);
			writePlant.then( function(value){
				remoteWriteStream = value;
				//console.log(value);
			});

			uploadPlantStream(localReadStream, remoteWriteStream);
			*/

			/*
			localReadStream.pipe(remoteWriteStream)
			  .on('error', function(err) {})
			  .on('finish', function() {
			  	
				console.log('Uploaded to bucket');
			    // The file upload is complete.

			    const results = [];

			    pg.connect(conString, (err, client, done) => {
			    	
			    	if(err){
			    		done();
			    		console.log(err);
			    	}
			*/
					//let username = "plantguy1";
			    	//let date = new Date();

			    	
		    		/*id SERIAL,
					plantID integer,
					plantType varchar(50),
					imgId integer,
					inputImg varchar(50),
					inputImgDir varchar(50),
					outputImg varchar(50),
					outputImgDir varchar(50),
					day integer,
					date date,
					growthCycle varchar(20),
					event varchar(20),
					width numeric,
					height numeric*/

			    	/*client.query(
			    		"INSERT INTO plants (plantID, plantType, growthCycle,\
			    		imgId, inputImg, inputImgDir, outputImg, outputImgDir, \
			    		day, date, event, \
			    		width, height) \
			    		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
			    	[plantID, plantType, growthCycle, 
			    	imgId, inputImg, inputImgDir, outputImg, outputImgDir, 
			    	day, date, event,
			    	width, height]);
					*/


					// TEST VERSION, IT WORKED

					//let SQL = "INSERT INTO test2 (username, date) VALUES ($1, $2)", [username, date];
			    	
			    	/*
					client.query("INSERT INTO test2 (username, date) VALUES ($1, $2)", [username, date]);
			    	*/

			    	/*const query = client.query("select * from test2");

			    	query.on('row', (row) => {
			    		results.push(row);
			    	});

			    	query.on('end', () => {
			    		done();
			    		console.log(JSON.stringify(results));
			    	});*/

			    //});

			  //});
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

    var form = new formidable.IncomingForm();

    var tmpfile = '';


    form.parse(req, function (err, fields, files) {
	   	
	   	//console.log(fields);
		//console.log(Object.keys(fields).length);


	   	if(Object.keys(fields).length !== 0){
	   		console.log("have file fields\n");
	   		var ifile = gcs.bucket(fields.bucket).file(fields.filename); 
	   		
	   		var local_ifile = './images/' + fields.filename;

	   		ifile.createReadStream()
	   			.pipe(fs.createWriteStream(local_ifile))
	   			.on('error', function(err) {})
	   			.on('response', function(response) {})
	   			.on('end', function(){})
	   			.on('finish', function(){
	   				console.log("have ifile\n");
	   			});
	   	} else {

			//console.log(files);

			var oldPath = files.filetoupload.path;
			var newPath = './images/';
			var filePath = newPath + files.filetoupload.name;


			fs.rename(oldPath, filePath, function (err) {

				if (err) throw err;

				imageExists = true;
				console.log("Exists in rename: " + imageExists);

				//fpath = oldPath;
				fpath = filePath;
				imgFile = files.filetoupload.name;
				pyTest();

				res.write('File uploaded!');
				res.end();

			});

		}
    });
	
    //readpath = tmpfile;

    //console.log('readpath: ' + readpath + '\n');


  imageExists = true;
  //pyTest();

  //res.write('File uploaded!');
  //res.end();

  } else {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write('<form action="fileupload" method="post" enctype="multipart/form-data">');
    res.write('<input type="file" name="filetoupload"><br>');
    res.write('<input type="submit">');
    res.write('</form>');
    return res.end();
  }
}).listen(3000); 

/*
	C14399846
	Oleg Petcov



*/


"use strict";

//https://www.w3schools.com/nodejs/nodejs_uploadfiles.asp
var http = require('http');
var formidable = require('formidable');
var fs = require('fs');
var ps = require('python-shell');
var gcloud = require('google-cloud');
const ChartjsNode = require('chartjs-node');


// Image and Python file data
var scriptLoc = '../pythonCode/imageProcess.py';
let imageExists = false;
var fpath = '';
var imgFile = null;


// Google Cloud Storage access
var gcs = gcloud.storage({
  projectId: 'image-processing-189813',
  keyFilename: './gcs/keyfile.json'
});


// Reference Image Buckets
var inBucketName = 'fypautobotany';
var outBucketName = 'fypautobotanyoutput'
var inputBucket = gcs.bucket(inBucketName);
var outputBucket = gcs.bucket(outBucketName);                

// Express
const express = require('express');
const app = express();


// Postges access details
const pg = require('pg');
const conString = "postgresql://postgres:imageskydatabase@35.205.117.19:5432/postgres";
const connectionInfo = {
	host : "35.205.117.19",
	port : 5432,
	database : "postgres",
	user: "postgres",
	password: "imageskydatabase"
};


// Streams for piping files
/*
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
*/


// Executes Python and Database code
function processImage(){

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

			// If there is not width or height, means there was no QR Code detected
			if (!width && !height) {
				console.log("Null or Empty measurements");
				width = -1;
				height = -1;
			}

			
			var contFileLoc = fileDir + plantContoursFilename;
			
			var bucketImg = plantContoursFilename;

			// Upload processed image to the Output Bucket
			outputBucket
				.upload(contFileLoc)
				.then( () => {
					
					console.log(`Uploaded output file ${contFileLoc} to ${outBucketName}`);
					
					
					// Begin Inserting data into the Postgres database
					pg.connect(conString, (err, client, done) => {
	
				    	if(err){
				    		console.log(err);
				    		done();
				    	}

				    	console.log("Connected to DB");
						
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

						// After finished Inserting,
						// Try deleting the locally storaged images 
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


// HTTP Server
http.createServer(function (req, res) {

  if (req.url == '/fileupload') {
  	
	console.log("Server accessed from: ");

    var form = new formidable.IncomingForm();

    var tmpfile = '';

    form.parse(req, function (err, fields, files) {

    	// The bucket upload portion of code
		// Cloud Function will send data about the Input image
	   	if(Object.keys(fields).length !== 0){

	   		console.log("\t bucket upload");

			// The File stored on the Input Bucket
	   		var ifile = gcs.bucket(fields.bucket).file(fields.filename); 
	   		
	   		var local_path = '../images/';

	   		var local_imgfile = local_path + fields.filename;

	   		ifile.createReadStream()
	   			.pipe(fs.createWriteStream(local_imgfile))
	   			.on('error', function(err) {})
	   			.on('response', function(response) {})
	   			.on('end', function(){})
	   			.on('finish', function(){
					
	   				console.log("Finished piping imgfile\n");
	   				console.log(local_imgfile);

	   				fpath = local_imgfile;
					
	   				imgFile = fields.filename;

					// Python Code
	   				processImage();

	   				res.writeHead(200, {Location:'http://35.189.222.193:3000/'});
					//res.status(200).redirect('http://35.189.222.193:3000/');
					//res.write('File uploaded!');
					res.end();
	   			});
	   	}

	   	// The GUI upload portion of the code (from browser)
		// Will send image to the Compute Server and Input Bucket
	   	else {

			console.log("\t browser");

			var oldPath = files.filetoupload.path;
			
			/*if (!oldPath){
	   			res.status(400).redirect('http://35.189.222.193:3000/');
	   		}*/

			var newPath = '../images/';
			
			var filePath = newPath + files.filetoupload.name;

			// Save uploaded image to disk
			fs.rename(oldPath, filePath, function (err) {

				if (err) {
					console.log(err);
					res.writeHead(301, {Location:'http://35.189.222.193:3000/'});
					res.end();
					//res.status(400).redirect('http://35.189.222.193:3000/');
				} 

				// Upload file to file Input Bucket
				inputBucket
					.upload(filePath)
					.then( () => {
						console.log(`Uploaded input file ${filePath} to ${inBucketName}`);
					});
				
				res.writeHead(200, {Location:'http://35.189.222.193:3000/'});
				//res.status(200).redirect('http://35.189.222.193:3000/');
				//res.write('File uploaded!');
				res.end();
	 			
			});
		}
    });

  imageExists = true;

  } 


  else if (req.url == '/graph') {

	pg.connect(conString, (err, client, done) => {

    	if(err){
    		console.log(err);
    		done();
    	}

    	console.log("Graphing: Connected to DB");
		
    	let firstday = 6;
		let lastday = -1;

		const results = [];

		try {
			const query = client.query("SELECT inputimg, ROUND(width,2) AS \"Width\", ROUND(height,2) AS \"Height\" FROM plants WHERE inputimg like 'day%' ");

			query.on('row', (row) => {
				results.push(row);
			});

			query.on('end', () => {
				done();
				const jsonStrArr = JSON.stringify(results, null, 1);

				var chartNode = new ChartjsNode(600, 600);

				var labelsArr = [];
				var wArr = [];
				var hArr = [];

				var jsonObjArray = JSON.parse(jsonStrArr);

				for (var i = 0; i < jsonObjArray.length; i++) {

					var str = jsonObjArray[i].inputimg;

					labelsArr.push(str.substring(0, str.length - 4));
					wArr.push(jsonObjArray[i].Width);
					hArr.push(jsonObjArray[i].Height);
				}


				var chartData = {
					labels: labelsArr,
					datasets: [{
							data: wArr,
							label: "Width",
							//backgroundColor: "rgba(153,255,51,0.4)",
							//borderColor: "#3e95cd",
							backgroundColor: "rgba(179,11,198,.2)",
  							borderColor: "rgba(179,11,198,1)",
							fill: false
						}, {
							data: hArr,
							label: "Heigth",
							//backgroundColor: "rgba(255,153,0,0.4)",
							//borderColor: "#8e5ea2",
							backgroundColor: "rgba(255,153,0,0.4)",
     						borderColor: "rgba(255,153,0,1)",
							fill: false
						}
					 ]
				};

				var chartOptions = {
					legend: {
						labels: {
							fontColor: '#ffffff'
							//fontColor: '#000'
						}
					},
					title: {
						display: true,
						text: 'Image Width and Height'
					},
					scales: {
					    yAxes: [{
					      scaleLabel: {
					        display: true,
					        labelString: 'CentiMetre'
					      }
					    }],
					    xAxes: [{
					      scaleLabel: {
					        display: true,
					        labelString: 'DAYS'
					      }
					    }],
					  }
				};

				var chartJsOptions = {
				    type: 'line',
				    data: chartData,
				    options: chartOptions
				};

				
				return chartNode.drawChart(chartJsOptions)
				.then(() => {
				    // chart is created
				 
				    // get image as png buffer
				    return chartNode.getImageBuffer('image/png');
				})
				.then(buffer => {
				    Array.isArray(buffer) // => true
				    // as a stream
				    return chartNode.getImageStream('image/png');
				})
				.then(streamResult => {
				    // using the length property you can do things like
				    // directly upload the image to s3 by using the
				    // stream and length properties
				    streamResult.stream // => Stream object
				    streamResult.length // => Integer length of stream
				    // write to a file
				    return chartNode.writeImageToFile('image/png', './testimage.png');
				})
				.then(() => {
					console.log("testChart file");
				    // chart is now written to the file path
				    // ./testimage.png


				    var fileToLoad = fs.readFileSync('./testimage.png');
				    res.writeHead(200, {'Content-Type': 'image/png'});
				    res.write(fileToLoad);
				    //res.write('<body>');
				    //res.write('<img src="./testimage.png" alt="Growth Over Days" width="1280" height="720">');
					//res.write('</body>');
				    return res.end();


				    // Returns JSON string array of width + height data
				    //res.write(jsonStrArr);
					//res.end();
				});

				// Returns JSON string array of width + height data
				//res.write(jsonStrArr);
				//res.end();
			});

		} catch (error){
			console.log(error);
		}

	});
	// HTML Form for browser access
	
    /*res.writeHead(200, {'Content-Type': 'text/html'});
    res.write('<center>');
    res.write('<form action="fileupload" method="post" enctype="multipart/form-data">');
    res.write('<input type="file" name="filetoupload" accept="image/png, image/jpg. image/jpeg, image/bmp"><br>');
    res.write('<input type="submit">');
    res.write('</form>');
	res.write('</center>');
    return res.end();*/
  }

  else {
	  
	// HTML Form for browser access
    var htmlFile = fs.readFileSync('./pages/home.html', {encoding: "utf8"});
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write(htmlFile);


    /*
    res.write('<center>');
    res.write('<form action="fileupload" method="post" enctype="multipart/form-data">');
    res.write('<input type="file" name="filetoupload" accept="image/png, image/jpg. image/jpeg, image/bmp"><br>');
    res.write('<input type="submit">');
    res.write('</form>');
	res.write('</center>');
	*/
    return res.end();
  }
}).listen(3000);

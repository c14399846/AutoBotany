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
var bucket = gcs.bucket('fypautobotanyoutput');                


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


function nJSSucksAss(callback,path){
     fs.readFile(path,'utf-8', function(err, data){
        if(err) return callback(err);
        //console.log(data);
	callback(null,data);
        //console.log(textData);
      });
}

function pyTest(){
	console.log("\nExists: " + imageExists + "\n");
	if(imageExists){

		var options = {
			mode: 'text',
			args: [fpath]	
		};
		console.log('fpath before python: ' + fpath);
		console.log(options);

		//var pyshell = new ps(scriptLoc);
		//console.log(imageExists);
		ps.run(scriptLoc, options, function(err, results){
			if(err) throw err;
			console.log('Finished py script\n');
			imageExists = false;
			console.log(results);
			var localReadStream = fs.createReadStream('./images/pContours.png');
			var bucketImg = 'pContours' + '_' + imgFile;
			var remoteWriteStream = bucket.file(bucketImg).createWriteStream();
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

					let username = "plantguy1";
			    	let date = new Date();

			    	//let SQL = "INSERT INTO test2 (username, date) VALUES ($1, $2)", [username, date];
			    	


			    	client.query("INSERT INTO test2 (username, date) VALUES ($1, $2)",
			    	[username, date]);

			    	const query = client.query("select * from test2");

			    	query.on('row', (row) => {
			    		results.push(row);
			    	});

			    	query.on('end', () => {
			    		done();
			    		console.log(JSON.stringify(results));
			    	});

			    });

			    /*massive(connectionInfo).then(instance => {
			    	console.log("\ndatabase thing\n");
			    	app.set("db", instance);

			    	// NEED TO ADD RETURN CRAP HERE
			    	let plantID = 1;
			    	let width = 1;
			    	let height = 1;
			    	let filename = "1";
			    	let day = "1";
			    	//let date = "1";
			    	let whateverElse = "1";

			    	let username = "plantguy1";

			    	let date = new Date();

			    	let SQL = "INSERT INTO public.test2 (username, date) VALUES (${testUsername}, ${$testDate})";
			    	app.get("db")
			    	  .query(SQL, {testUsername:username}, {testDate:date})
			    	  .catch(error => console.log("Failed to upload data to Database"));
			    });*/

			  });
		});

	}
}


//var upload_path = "/home/image/node/";

http.createServer(function (req, res) {
  if (req.url == '/fileupload') {
    var form = new formidable.IncomingForm();

    var tmpfile = '';


    form.parse(req, function (err, fields, files) {
	   	
	   	console.log(fields);
		console.log(Object.keys(fields).length);


	   	if(Object.keys(fields).length !== 0){
	   		console.log("have file fields\n");
	   		var ifile = gcs.bucket(fields.bucket).file('day19.png'); 
	   		
	   		var local_ifile = './images/' + 'day19.png';

	   		ifile.createReadStream()
	   			.on('error', function(err) {})
	   			.on('response', function(response) {})
	   			.on('end', function(){})
	   			.on('finish', function(){})
	   			.pipe(fs.createWriteStream(local_ifile));

	   		//console.log(local_ifile);
	   	} else {

	   	//console.log("\n\n\n\n");
		//console.log(req.body);

		//console.log(req.url);
		//console.log(req);
		//console.log("\n\n\n\n");
		console.log(files);

		var oldPath = files.filetoupload.path;
		//var newpath = upload_path + files.filetoupload.name;
		//var newPath = '/home/image/images/';
		var newPath = './images/';
		var filePath = newPath + files.filetoupload.name;


		fs.rename(oldPath, filePath, function (err) {
			if (err) throw err;

			//pyshell.run();

			imageExists = true;
			console.log("Exists in rename: " + imageExists);

			//fpath = oldPath;
			fpath = filePath;
			imgFile = files.filetoupload.name;
			pyTest();
			/*
			var localReadStream = fs.createReadStream('/home/image/images/pcontours.png');
			var remoteWriteStream = bucket.file('pcontours.png').createWriteStream();
			localReadStream.pipe(remoteWriteStream)
			  .on('error', function(err) {})
			  .on('finish', function() {
			    // The file upload is complete.
			  });
			*/

			res.write('File uploaded!');
			res.end();

		});

		}

		/*
		tmpfile = files.filetoupload.path;
		console.log('tmpfile: ' + files.filetoupload.path);


		fs.readFileSync(tmpfile, function(err,data){
			if(!err){
				console.log(data);
			}
			else {
				console.log("error\n");		
			}	
		});
		*/







		//res.write('File uploaded!');
	  	//res.end();

		/*
		var oldPath = files.filetoupload.path;
		//var newpath = upload_path + files.filetoupload.name;
		var newPath = '/home/image/images/';

		var filePath = newPath + files.filetoupload.name;
		*/

		//var readPath = files.fileupload.path.toString();
		//var textData = nJSSucksAss(function(err, readPath){console.log(readPath);});
		/*fs.readFile(files.fileupload.path, function(err, data){
		var array = data.toString();
		console.log(array);
		});*/
		  /*fs.writeFile(filePath, textData, function(err){
			if(err){return console.log(err);}
				console.log(files.fileupload.name);
			console.log("File saved");
		});*/
		//res.write('File uploaded');
		//res.end();

		/*fs.rename(oldPath, filePath, function (err) {
		if (err) throw err;

		//pyshell.run();

		imageExists = true;
		console.log("Exists in rename: " + imageExists);

		//res.write('File uploaded!');
		//res.end();

		});*/

		
		//console.log("Exists before Python: " + imageExists);

		/*if(imageExists){
			console.log(filePath);
			fpath = oldPath;
			pyTest();
		}*/
		//console.log('filePath:' + filePath);


		/*
		fpath = oldPath + '.PNG';
		
		readpath = fpath;

		console.log('fpath: ' + fpath + '\n');
		fs.readFileSync(readpath, function(err,data){
			if(!err){
				console.log(data);
			}
			else {
				console.log("error\n");		
			}	
		});

		imageExists = true;
		//pyTest();

		res.write('File uploaded!');
	        res.end();
		*/
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

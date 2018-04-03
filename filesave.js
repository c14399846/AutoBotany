//https://www.w3schools.com/nodejs/nodejs_uploadfiles.asp
var http = require('http');
var formidable = require('formidable');
var fs = require('fs');
var ps = require('python-shell');


//var scriptLoc = './../cs.py';
var scriptLoc = './../imageProcess.py';
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
			var localReadStream = fs.createReadStream('/home/image/images/pContours.png');
			var bucketImg = 'pContours' + '_' + imgFile;
			var remoteWriteStream = bucket.file(bucketImg).createWriteStream();
			localReadStream.pipe(remoteWriteStream)
			  .on('error', function(err) {})
			  .on('finish', function() {
				console.log('Uploaded to bucket');
			    // The file upload is complete.
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
		//console.log(files.fileupload);
		//console.log(files);

		var oldPath = files.filetoupload.path;
		//var newpath = upload_path + files.filetoupload.name;
		var newPath = '/home/image/images/';
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

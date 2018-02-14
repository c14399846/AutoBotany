//https://www.w3schools.com/nodejs/nodejs_uploadfiles.asp
var http = require('http');
var formidable = require('formidable');
var fs = require('fs');
var ps = require('python-shell');


var scriptLoc = './../cs.py';
var pyshell = new ps(scriptLoc);
var imageExists = false;


function nJSSucksAss(callback,path){
     fs.readFile(path,'utf-8', function(err, data){
        if(err) return callback(err);
        //console.log(data);
	callback(null,data);
        //console.log(textData);
      });


}

function pyTest(){
	if(imageExists){
	console.log(imageExists);
		ps.run(scriptLoc, function(err){
			if(err) throw err;
			console.log('Finished py script\n');
			imageExists = false;
		});
	}
}


//var upload_path = "/home/image/node/";

http.createServer(function (req, res) {
  if (req.url == '/fileupload') {
    var form = new formidable.IncomingForm();
    form.parse(req, function (err, fields, files) {
	//console.log(files.fileupload);
	//console.log(files.fileupload.name);
	//console.log(files.fileupload.path);   
	var oldPath = files.fileupload.path;
	//var newpath = upload_path + files.filetoupload.name;
	var newPath = '/home/image/images/';

      var filePath = newPath + files.fileupload.name;

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
	
      fs.rename(oldPath, filePath, function (err) {
        if (err) throw err;
	
	//pyshell.run();
	
        res.write('File uploaded!');
        res.end();
	imageExists = true;
      });
	console.log(imageExists);
	if(imageExists){
		console.log(filePath);
		pyTest();
	}

 });
  } else {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write('<form action="fileupload" method="post" enctype="multipart/form-data">');
    res.write('<input type="file" name="fileupload"><br>');
    res.write('<input type="submit">');
    res.write('</form>');
    return res.end();
  }
}).listen(3000); 

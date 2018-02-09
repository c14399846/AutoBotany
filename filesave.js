https://www.w3schools.com/nodejs/nodejs_uploadfiles.asp

var http = require('http');
var formidable = require('formidable');
var fs = require('fs');

http.createServer(function (req, res) {
  if (req.url == '/fileupload') {
    var form = new formidable.IncomingForm();
    form.parse(req, function (err, fields, files) {
      //var oldpath = files.filetoupload.path;
      
	  var filePath = '~/image/' + files.fileupload.name;
	  
	  var fData = [];
	  
	  fs.readFile(files.fileupload.path, function(err, data){
		  if(err) throw err;
		  
		  console.log(data);
		  fData = data;
	  });
	  
	  
	  fs.writeFile(filePath, "hello", function(err){
		if (err){return console.log(err);}
		
		console.log(fiels.fileupload.name)
		console.log("File Saved");
		
		res.write("File Uploaded");
		res.end();		
	  });
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

var Curl = require('./node_modules/node-libcurl/lib/Curl'),
  path = require('path'),
  fs = require('fs');

var curl = new Curl(),
  url = 'http://127.0.0.1:3000/fileupload',
  image = fs.readFile('/home/image/images/PEA_16.PNG'),
  imageFilename = path.resolve('/home/image/images/', 'PEA_16.PNG'),
  data = [
    {
      name: 'PEA_16.PNG',
      contents: 'Pea Image',
    },
    {
      name: 'filetoupload',
      file: imageFilename,
      type: 'image/png',
    },
  ];

curl.setOpt(Curl.option.URL, url);
curl.setOpt(Curl.option.HTTPPOST, data);
curl.setOpt(Curl.option.VERBOSE, true);

curl.perform();

curl.on('end', function(statusCode, body) {
  console.log(body);

  this.close();
  //fs.unlinkSync(imageFilename);
});

curl.on('error', function() {
  this.close();
  //fs.unlinkSync(imageFilename);
});

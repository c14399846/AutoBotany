/**
 * @author Jonathan Cardoso Machado
 * @license MIT
 * @copyright 2015, Jonathan Cardoso Machado
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is furnished to do
 * so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/**
 * Example showing how to upload a file using node-libcurl
 * The upload is going to be the same than one done using POST and a multipart/form-data
 */
var Curl = require('./node_modules/node-libcurl/lib/Curl'),
  path = require('path'),
  fs = require('fs');

var curl = new Curl(),
  //url  = 'http://posttestserver.com/post.php',
  url = 'http://127.0.0.1:3000/fileupload',
  image = fs.readFile('/home/image/images/PEA_16.PNG'),
  imageFilename = path.resolve('/home/image/images/', 'PEA_16.PNG'),
  //buff = new Buffer(image, 'base64'),
  data = [
    {
      name: 'PEA_16',
      contents: 'Pea Image',
    },
    {
      name: 'filetoupload',
      file: imageFilename,
      type: 'image/png',
    },
  ];

//create the image file
//fs.writeFileSync(imageFilename, buff);

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

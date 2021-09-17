//Load a text resource from a file over the internet
var loadTextResource = function (url, callback) {
	var request = new XMLHttpRequest();
	request.open('GET', url + '?please-dont-cache=' + Math.random(), true);
	request.onload = function ()
    {
		if (request.status < 200 || request.status > 299)
        {
			callback(true,'Error: HTTP Status ' + request.status + ' on resource ' + url);
		} else {
			callback(false, request.responseText);
		}
    };
	request.send();
};


//Download Section
function GridSize(data) {
    var Grid_Size = 21; //Going to be one greater if doesn't divid fully
    var x_ratio = Math.floor(x_size/Grid_Size);
    var y_ratio = Math.floor(y_size/Grid_Size);
    var my_data = [];    
    for (y=0; y<y_size; y=y+y_ratio) {
        for (x=0; x<x_size; x=x+x_ratio)
        {
        my_data.push(data[x+y*x_size]);
        }
    }
    return my_data;
}


function download_data(all_data) {
    var my_text = '';
    
    for (let row = 0; row < all_data.length; row++)
    {
    //Store each grid one line
        for (let i = 0; i < all_data[row].length; i++) {
            my_text += all_data[row][i] + ",";
        }
        my_text += "\n";
    }
    
    download("Test.txt", my_text);
}

function download(filename, text) {
  var element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  element.setAttribute('download', filename);

  element.style.display = 'none';
  document.body.appendChild(element);

  element.click();

  document.body.removeChild(element);
}
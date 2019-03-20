const CLASSES =  ({0:'angry',1:'disgust',2:'fear',3:'happy', 4:'sad',5:'surprise',6:'neutral'});
var originalVideoWidth=640
let emotionModel;
let emotion= [6,CLASSES[6],1.000000];


// load model
async function loadFaceModel(){
    console.log("model loading..");
    emotionModel = await tf.loadModel (`./models/emotion/model.json`);
    console.log("model loaded.");
};

async function faceTracker(ctx){
    const tracker = new tracking.ObjectTracker('face');
    tracker.setInitialScale(4);
    tracker.setStepSize(2);
    tracker.setEdgesDensity(0.1);
    tracking.track("#video", tracker, { camera: true });
    tracker.on('track', function(event) {
        ctx.clearRect(0, 0, contentWidth, contentWidth);
        event.data.forEach( function (rect) {
            if (!event.data) return;
            predictEmotion(rect);
           // console.log(emotion[0],emotion[1],emotion[2]);
            if(emotion[0]=3&&emotion[2]>0.8){
                actions[1].stop();
                actions[0].play();
            }else {
                actions[0].stop();
                actions[1].play();
            }
            //console.log('x: ' + rect.x + 'px', rect.x + rect.width + 5, rect.y + 11);
            //console.log('y: ' + rect.y + 'px', rect.x + rect.width + 5, rect.y + 22);
        });
    });
}

async function predictEmotion(rect){

    let tensor = captureWebcam(rect) ;

    let prediction = await emotionModel.predict(tensor).data();
    let results = Array.from(prediction)
        .map(function(p,i){
            return {
                probability: p,
                className: CLASSES[i],
                classNumber: i
            };
        }).sort(function(a,b){
            return b.probability-a.probability;
        }).slice(0,6);

    results.forEach(function(p){
        return emotion = [results[0].classNumber,results[0].className, results[0].probability]
    });
};


// capture streaming video
function captureWebcam(rect) {
    var faceCanvas = document.getElementById('faceCanvas');
    var faceContext = faceCanvas.getContext('2d');

    //adjust original video size
    var adjust = originalVideoWidth/contentWidth;
    faceContext.drawImage(video, rect.x * adjust , rect.y * adjust, rect.width * adjust, rect.height * adjust,0, 0, 100, 100);

    tensor_image = preprocessImage(faceCanvas);

    return tensor_image;
}

//-----------------------
// TensorFlow.js method
// image to tensor
//-----------------------

function preprocessImage(image){
    const channels = 1;

    let tensor = tf.fromPixels(image, channels).resizeNearestNeighbor([64,64]).toFloat();
    let offset = tf.scalar(255);
    return tensor.div(offset).expandDims();
};

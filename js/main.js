const imageScaleFactor = 0.2;
const outputStride = 16;
const stats = new Stats();
const modelScale=0.02
const contentWidth = 800;
const contentHeight = 600;
let renderer, scene, camera,mixer,video;
let pikaModel;
let loading;
let clock = new THREE.Clock();
let actions = [];
let ctx;

// video属性のロード
async function loadVideo() {
    const video = await setupCamera(); // カメラのセットアップ
    video.play();
    return video;
}

// カメラのセットアップ
// video属性からストリームを取得する
async function setupCamera() {
    video = document.getElementById('video');
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({
            'audio': false,
            'video': true});
        video.srcObject = stream;

        return new Promise(resolve => {
            video.onloadedmetadata = () => {
                resolve(video);
            };
        });
    } else {
        const errorMessage = "This browser does not support video capture, or this device does not have a camera";
        alert(errorMessage);
        return Promise.reject(errorMessage);
    }
}

// 取得したストリームをestimateSinglePose()に渡して姿勢予測を実行
// requestAnimationFrameによってフレームを再描画し続ける
function detectPoseInRealTime(video, net) {
    const canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    const flipHorizontal = true; // since images are being fed from a webcam
    ctx.scale(-1, 1);
    loadFaceModel()
    //faceTracker(ctx);
    async function poseDetectionFrame() {
        requestAnimationFrame(poseDetectionFrame);
        let poses = [];
        const pose = await net.estimateSinglePose(video, imageScaleFactor, flipHorizontal, outputStride);
        poses.push(pose);
        ctx.clearRect(0, 0, contentWidth,contentHeight);
        ctx.save();

        ctx.translate(-contentWidth, 0);
        ctx.drawImage(video, 0, 0, contentWidth, contentHeight);
        ctx.restore();

        poses.forEach(({ score, keypoints }) => {
            updateModel(keypoints[5],keypoints[6]);
        });
        renderer.render( scene, camera );
        var time = clock.getDelta();
        if (mixer) {
            mixer.update(time);
        }
    }
    poseDetectionFrame();



}

//THREEのレンダラの初期化
const initRenderer = async () => {
    //z-fighting対策でlogarithmicDepthBufferを指定
    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, logarithmicDepthBuffer: true });
    renderer.gammaOutput = true;
    renderer.setClearColor(new THREE.Color(0xffffff), 0);
    renderer.setSize(contentWidth, contentHeight);
    renderer.domElement.style.position = "absolute";
    document.getElementById("right1").appendChild(renderer.domElement);
}
//THREEのシーンの初期化
const initScene = async () => {
    //シーン
    scene = new THREE.Scene();
    //カメラ
    camera = new THREE.PerspectiveCamera(45, contentWidth / contentHeight, 1, 1000 );
    camera.position.set(0, 0, 0);
    scene.add(camera);
    //ライト
    let light = new THREE.AmbientLight(0xffffff, 1.0);
    scene.add(light);
    //モデルの読み込み
    let result = await load3DModel();

    return result;
}

const load3DModel = async () => {
    var loader = new THREE.FBXLoader();
    let result = await new Promise(resolve => {
        loader.load('assets/pika2.fbx', function (object) {
            object.position.set(0, -0.5, -2);
            object.scale.set(modelScale,modelScale,modelScale);
            mixer = new THREE.AnimationMixer(object);
            actions[0]=mixer.clipAction(object.animations[117]);
            actions[1]=mixer.clipAction(object.animations[94]);

            actions[1].play();
            object.traverse(function (child) {

                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                }

            });
            scene.add(object);
            pikaModel=object;
            return resolve(object);
        });
    });
    return result;
}
function updateModel(keyPoint1,keyPoint2) {
    var disOfShoulder=distanceVector(keyPoint1.position,keyPoint2.position);
    var rate=disOfShoulder/600;
    let pX = ((keyPoint1.position.x) / 800) * 2 - 1+0.3*rate;
    let pY = - ((keyPoint1.position.y) / 600) * 2 + 1+0.4*rate;
    pikaModel.position.set(pX,pY,-2);
    pikaModel.scale.set(modelScale*rate,modelScale*rate,modelScale*rate);
    pikaModel.lookAt(0,0,0);
}

function distanceVector( v1, v2 )
{
    var dx = v1.x - v2.x;
    var dy = v1.y - v2.y;
    return Math.sqrt( dx * dx + dy * dy  );
}

async function bindPage() {
    const net = await posenet.load(); // posenetの呼び出し
    let video;
    try {
        video = await loadVideo(); // video属性をロード
    } catch(e) {
        console.error(e);
        return;
    }
    const resRenderer = initRenderer();
    const resScene = initScene();
    await Promise.all([resRenderer, resScene]);


    detectPoseInRealTime(video, net);

}

function clearLoading() {
    faceTracker(ctx);
    loading = document.getElementById("loading");
    loading.style.display = "none";
}
bindPage();
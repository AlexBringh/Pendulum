import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';  
import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';
import WebGL from 'three/addons/capabilities/WebGL.js';


// Import data from files



// Some variables for user control of the animation.
let isStarted = false;
let isPaused = false;

// Reference objects for the buttons and dropdown selections.
let startButton  = document.getElementById("start_animation_btn");
let pauseButton  = document.getElementById("pause_animation_btn");
let stopButton   = document.getElementById("stop_animation_btn");
let resetCamera  = document.getElementById("reset_camera_btn");
let dataDropdown = document.getElementById("datapoints_dropdown");

// Setup button controls
startButton.addEventListener("click", function () {
    /**
     * Start animation.
     * Button event listener to start the animation.
     */
    isStarted = true;
    isPaused = false;
    startButton.disabled = "disabled";
    pauseButton.disabled = "";
    stopButton.disabled = "";
    });

pauseButton.addEventListener("click", function() {
    /**
     * Pause animation.
     * Butten event listener to pause the animation.
     */
    isPaused = true;
    startButton.disabled = "";
    pauseButton.disabled = "disabled";
    stopButton.disabled = "";
});

stopButton.addEventListener("click", function() {
    /**
     * Stop animation.
     * Button event listener to stop the animation.
     */
    isStarted = false;
    isPaused = false;
    reset_model();
    startButton.disabled = "";
    pauseButton.disabled = "disabled";
    stopButton.disabled = "disabled";
});

resetCamera.addEventListener("click", function () {
    /**
     * Reset camera button.
     * Button event listener to reset the camera position for the viewer.
     */
    camera.position.set(1000,1000,0)
});

// Setup event listener for dropdown selection of datapoints to animate.
dataDropdown.addEventListener("change", function() {

});


/**
 * Setup for WebGL animation through the THREE.js library.
 */

// Create the scene and the renderer
const scene = new THREE.Scene();
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerHeight, window.innerHeight / 1.5);
renderer.setClearColor(0x0000ff, 0.4);

// Attach the renderer to the screen.
const pageAnimationContainer = document.getElementById("animation_container");
pageAnimationContainer.appendChild(renderer.domElement);

// Create a camera that gives us a field of view to the scene. I don't know what all these ratios and variables do and nor do I want to know so long as it works.
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2500);
camera.position.set(1000, 1000, 0);
camera.lookAt(0, 1010, 0); // Makes sure we are looking at the right spot.

// Add some orbital controls because why not.
const controls = new OrbitControls(camera, renderer.domElement);
controls.minDistance = 1000;
controls.maxDistance = 2000;

// Lights so we can see anything on the screen.
const light = new THREE.AmbientLight( 0x404040 , 40 );
scene.add( light );


// Define the 3D Objects for the pendulum
const tower = new THREE.Object3D();
const pendelArm1 = new THREE.Object3D();
const pendelArm2 = new THREE.Object3D();
const pendelBall = new THREE.Object3D();

// Load .obj model files and corresponding .mtl texture files.
const mtl_loader = new MTLLoader();

mtl_loader.load("./model/tower.mtl", function (materials) {
    materials.preload();
    const obj_loader = new OBJLoader();
    obj_loader.setMaterials(materials);
    obj_loader.load("./model/tower.obj", function (object) {
        tower.add(object);
    });
});

mtl_loader.load("./model/pendel_arm_1.mtl", function (materials) {
    materials.preload();
    const obj_loader = new OBJLoader();
    obj_loader.setMaterials(materials);
    obj_loader.load("./model/pendel_arm_1.obj", function (object) {
        pendelArm1.add(object);
    });
});

mtl_loader.load("./model/pendel_arm_2.mtl", function (materials) {
    materials.preload();
    const obj_loader = new OBJLoader();
    obj_loader.setMaterials(materials);
    obj_loader.load("./model/pendel_arm_2.obj", function (object) {
        pendelArm2.add(object);
    })
});

mtl_loader.load("./model/pendel_ball.mtl", function (materials) {
    materials.preload();
    const obj_loader = new OBJLoader();
    obj_loader.setMaterials(materials);
    obj_loader.load("./model/pendel_ball.obj", function (object) {
        pendelBall.add(object);
    });
});

// Attach the parts together to form the complete model.
tower.add(pendelArm1);
pendelArm1.add(pendelArm2);
pendelArm2.add(pendelBall);

// Set the correct positions of the bodies.
tower.position.y = -200;
pendelArm1.rotation.x = Math.PI;
pendelArm1.position.y = 710;
pendelArm1.position.x = 50;
pendelArm2.rotation.x = Math.PI;
pendelArm2.position.z = 430;
pendelArm2.position.x = 30;

// Add the pendulum to the scene.
scene.add(tower);

// Make a check that WebGL is available to be used on the device seeing the site. 
if (WebGL.isWebGLAvailable()) {
    animate();
} else {
    alert("Error in loading WebGL, cannot proceed to animate the model.")
}

function animate()
{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);

    // TODO: Write code for animating with proper timesteps between each datapoint.
    if (!isPaused && isStarted) {

    }
}

function reset_model ()
{
    pendelArm1.rotation.x = Math.PI;
    pendelArm2.rotation.x = Math.PI;
}



// Present graph images corresponding to the animation

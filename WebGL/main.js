import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';  
import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';
import WebGL from 'three/addons/capabilities/WebGL.js';

// Variable to mark busy with importing new data so the animation can't start before the data is ready.
let isBusyImportingData = true;

// Datapoint arrays for the animation.
const time = []; // This is by default always empty because the animation doesn't actually need it.
const theta = [];
const phi = [];
const theta_dot = []; // This is by default always empty because the animation doesn't actually need it.
const phi_dot = [];   // This is by default always empty because the animation doesn't actually need it.
let timestep;

// Variable to keep track of where in the array we are for the animation.
let counter = 0;

// Import data from files
async function get_data(filename)
{
    // Mark as busy importing data.
    isBusyImportingData = true;

    // Just making sure that the counter variable is set back to 0.
    counter = 0;

    // Clear out the previous data from the arrays.
    time.length = 0;
    theta.length = 0;
    phi.length = 0;
    theta_dot.length = 0; 
    phi_dot.length = 0;

    // Import datapoints from the .csv file as text. In javascript, we have to treat it as pure text and parse it manually.
    const response = await fetch('data/' + filename);
    const data = await response.text();

    // Split into array where \n (newline) marks end of entry in array. Slice removes the first row, the title row.
    const rows = data.split('\n').slice(1);
    
    //console.log(rows);
    // Loop through the rows.
    rows.forEach(row_element => {
        if (row_element != "") {
            const row = row_element.split(',');
            // time.push(parseFloat(row[1])); // Not needed by the animation so is not included by default.
            theta.push(parseFloat(row[2]));
            phi.push(parseFloat(row[3]));
            // theta_dot.push(parseFloat(row[4])); // Not needed by the animation so is not included by default.
            // phi_dot.push(parseFloat(row[5])); // not needed by the animation sois not included by default.
            timestep = parseFloat(row[6]);
        }
    });
    isBusyImportingData = false;
}

// Import default data for animation.
get_data("RK4_man_NoAirResist, time02,08,18 date12,11,2023.csv");


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
    dataDropdown.disabled = "disabled";
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
    dataDropdown.disabled = "disabled";
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
    dataDropdown.disabled = "";
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
    switch (dataDropdown.value)
    {
        case "no_air_res_th0:0_ph0:0_td:0.01":
            get_data("RK4_man_NoAirResist, time02,08,18 date12,11,2023.csv");
            break;
        case "air_res_th0:0_ph:0_td:0.01":
            get_data("RK4_man_NoAirResist, time02,08,18 date12,11,2023.csv"); //TODO: This is temporary, change with actual datapoints later.
            break;
    }
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
pendelArm2.rotation.x = 0;
pendelArm2.position.z = 200;
pendelArm2.position.x = 30;
pendelBall.position.z = 230;

// Add the pendulum to the scene.
scene.add(tower);

// Make a check that WebGL is available to be used on the device seeing the site. 
if (WebGL.isWebGLAvailable()) {
    animate();
} else {
    alert("Error in loading WebGL, cannot proceed to animate the model.")
}

// Animate function for THREE.js WebGL animation implementation.
function animate()
{
    
    requestAnimationFrame(animate)
    controls.update();
    renderer.render(scene, camera);

    // TODO: Write code for animating with proper timesteps between each datapoint.
    
    if (!isPaused && isStarted && !isBusyImportingData) {
        pendelArm1.rotation.x = Math.PI + theta[counter];
        pendelArm2.rotation.x = phi[counter];
        counter += 1;
        if (counter >= theta.length) {
            counter = 0;
        }
    }
}

function reset_model ()
{
    pendelArm1.rotation.x = Math.PI;
    pendelArm2.rotation.x = 0;
}


// Present graph images corresponding to the animation

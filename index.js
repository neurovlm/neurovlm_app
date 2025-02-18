//import init_query, { read_npz_file, run_query } from './pkg/neuroquery.js';
import init_encoder, { read_npz_file_encoder, read_model_tensors, text_query } from './pkg_encoder/neuroencoder.js';
import {Niivue, NVImage} from "./js/niivue.min.js";
// run_inverse_query

async function run() {
    // Initialize the WASM module
    // await init_query();
    await init_encoder();

    // Encoder: small BERT
    const weights = await fetch('./pkg_encoder/assets/model.safetensors')
        .then(res => {
            if (!res.ok) {
                throw new Error(`Failed to fetch weights: ${res.status} ${res.statusText}`);
            }
            return res.arrayBuffer();
        })
        .catch(err => {
            console.error("Error fetching weights:", err);
            throw err;
        });
    const weights_array = new Uint8Array(weights);

    const citations = await fetch('./pkg_encoder/assets/titles.txt')
        .then(response => response.text())
        .then(data => {
            return data.split('\n');  // Split the file content into lines
        })
        .catch(error => console.error('Error loading file:', error));

    const citation_links = await fetch('./pkg_encoder/assets/links.txt')
        .then(response => response.text())
        .then(data => {
            return data.split('\n');  // Split the file content into lines
        })
        .catch(error => console.error('Error loading file:', error));

    const config_response = await fetch('./pkg_encoder/assets/config.json');
    const config_data = await config_response.json();
    const config = JSON.stringify(config_data);

    const tk_response = await fetch('./pkg_encoder/assets/tokenizer.json');
    const tokenizer = new Uint8Array(await tk_response.arrayBuffer());

    const titles = await fetch('./pkg_encoder/assets/titles.safetensors')
        .then(res => {
            if (!res.ok) {
                throw new Error(`Failed to fetch weights: ${res.status} ${res.statusText}`);
            }
            return res.arrayBuffer();
        })
    const titles_array = new Uint8Array(titles);

    const adj_response = await fetch('./pkg_encoder/assets/adjustments.json');
    const adj = await adj_response.json();

    const scrollTop = document.getElementById("output_container").scrollTop;

    // Fetch the neuroquery arrays
    fetch('./pkg/constants.npz')
        .then(response => response.arrayBuffer()) // Convert to ArrayBuffer
        .then(arrayBuffer => {
            const uint8Array = new Uint8Array(arrayBuffer); // Convert to Uint8Array
            try {
                // read_npz_file(uint8Array); // Pass to the Rust function
                read_npz_file_encoder(uint8Array, "surface");
                // read_npz_file_encoder(uint8Array);
            } catch (error) {
                console.error("Error reading .npz file:", error);
            }
        })
        .catch(error => {
            console.error("Failed to load constants.npz file:", error);
        });

    // Get text box input
    const textbox = document.getElementById("txid");

    // Neuroencoder tensors
    const aligner = await fetch('./pkg_encoder/assets/aligner.safetensors')
        .then(res => {
            if (!res.ok) {
                throw new Error(`Failed to fetch weights: ${res.status} ${res.statusText}`);
            }
            return res.arrayBuffer();
        })
        .catch(err => {
            console.error("Error fetching weights:", err);
            throw err;
        });
    const aligner_weights = new Uint8Array(aligner);

    const decoder = await fetch('./pkg_encoder/assets/decoder.safetensors')
        .then(res => {
            if (!res.ok) {
                throw new Error(`Failed to fetch weights: ${res.status} ${res.statusText}`);
            }
            return res.arrayBuffer();
        })
        .catch(err => {
            console.error("Error fetching weights:", err);
            throw err;
        });
    const decoder_weights = new Uint8Array(decoder);

    // Set global tensors
    read_model_tensors(
        weights_array,
        config,
        tokenizer,
        titles_array,
        adj,
        aligner_weights,
        decoder_weights
    );

    // Add event listener to capture input text on Enter key press
    textbox.addEventListener('keypress', function (event) {
        if (event.key === 'Enter' && this.value.trim() !== "") {

            let textQuery = [];

            // Prevent default action (like adding a new line)
            event.preventDefault();

            // Add the input text to the array
            textQuery.push(this.value);

            // // Neuroquery
            // var _surface_map = new run_query(textQuery[0]);
            // let surface_map = new Float32Array(327684);

            // Autoencoder query
            let out = text_query(this.value);

            let _surface_map = out.surface;
            let volume = out.volume;
            let pub_order = out.puborder;

            let surface_map = new Float32Array(327684);

            let ordered_citations = Array.from(pub_order).map(i => citations[Math.floor(i)]);
            let ordered_links =Array.from(pub_order).map(i => citation_links[Math.floor(i)]);

            let citations_container = document.getElementById("output_container");
            citations_container.innerHTML = ordered_citations.slice(0, 100).map(
                (item, index) => `<div class="spaced"><a href="${ordered_links[index]}" target="_blank"> [${index+1}] ${item} </a> </div>`
            ).join('');
            citations_container.scrollTop = scrollTop;

            // Re-order left
            fetch('remap.txt')
                .then(response => response.text())
                .then(data => {
                    const reinds = JSON.parse(data);
                    for (var i=0; i < 163842; i++){
                        surface_map[i] = _surface_map[reinds[i]];
                        surface_map[i+163842] = _surface_map[reinds[i+163842]];
                    }

                    // Visualize
                    let surface_min = surface_map.reduce((min, curr) => curr < min ? curr : min, Infinity);
                    let surface_max = surface_map.reduce((max, curr) => curr > max ? curr : max, -Infinity);

                    // Re-create dataview
                    dataviews = dataset.fromJSON({
                        "views": [{
                            "data": ["__e506186d71676e98"],
                            "state": null,
                            "attrs": {"priority": 1},
                            "desc": "",
                            "cmap": ["RdBu_r"],
                            "vmin": [surface_min],
                            "vmax": [surface_max],
                            "name": "neurovlm"
                        }],
                        "data": {
                            "__e506186d71676e98": {
                                "split": 163842,
                                "frames": 1,
                                "name": "__e506186d71676e98",
                                "subject": "fsaverage",
                                "min": surface_min,
                                "max": surface_max,
                                "raw": false
                            }
                        },
                        "images": {
                            "__e506186d71676e98": [surface_map]
                        }
                    });
                    viewer.addData(dataviews);

                    // Clear the textarea for new input
                    this.value = "";
                })
                .catch(error => console.error('Error loading the file:', error));

            // Volume
            const dims = [46, 55, 46];
            const pixDims = [1, 1, 1];
            const affine = [
                4., 0., 0., -90.,
                0., 4., 0., -126.,
                0., 0., 4., -72.,
                0., 0., 0., 1.
            ];

            const datatypeCode = 16; // code for float32
            console.log(volume);
            const slicedArray = volume.slice(0, 20000); // [3, 4, 5]
            const sum = slicedArray.reduce((acc, val) => acc + val, 0);
            console.log(sum)

            nv1.createNiftiArray(dims, pixDims, affine, datatypeCode, volume)
                .then(bytes => {
                    if (nv1.volumes[1] != null){
                        nv1.removeVolume(nv1.volumes[1]);
                    };
                    NVImage.loadFromUrl({
                        url: bytes,
                        colormap: "magma",
                        visible: true,
                        opacity: 0.5,
                        cal_min: 0.5,
                        cal_max: 1.0
                    }).then(nii => {
                        console.log(nii);
                        nv1.addVolume(nii);
                    });

                });
        }
    });

    // Neuromaps
    let dataSelection = document.getElementById("dataname");
    dataSelection.onclick = function() {
        // Re-create dataview
        fetch('data/data.json')
            .then(response => response.json())
            .then(data => {
                data.views[0].vmin = [0.0];
                data.views[0].vmax = [1.0];
                data.data.__e506186d71676e98.min = 0.0;
                data.data.__e506186d71676e98.max = 1.0;
                dataviews = dataset.fromJSON(data);
                viewer.addData(dataviews);
          });
    };

    // MNI Space
    const canvas = document.createElement('canvas');
    canvas.id = 'gl1';
    canvas.style.backgroundColor = "#131314";

    let toggleContainer = document.createElement("div");
    toggleContainer.className = "toggle-container"

    let mniDiv = document.createElement("div");
    let fsaverageDiv = document.createElement("div");

    mniDiv.className = "toggle-option";
    fsaverageDiv.className = "toggle-option active";
    mniDiv.id = "mni";
    fsaverageDiv.id = "fsaverage";
    mniDiv.textContent = "MNI";
    fsaverageDiv.textContent = "fsaverage";

    toggleContainer.appendChild(fsaverageDiv);
    toggleContainer.appendChild(mniDiv);
    document.body.appendChild(toggleContainer);

    let parent = document.getElementsByName("w2figure0")[0]
    parent.appendChild(canvas);

    var drop = document.getElementById("sliceType");
    drop.onchange = function () {
        let st = parseInt(document.getElementById("sliceType").value);
        nv1.setSliceType(st);
    }
    function handleIntensityChange(data) {
        document.getElementById("intensity").innerHTML =
        "&nbsp;&nbsp;" + data.string;
    }
    var volumeList1 = [
        {
            url: "./mni/mni152.nii.gz",
            colormap: "gray",
            visible: true,
            opacity: 1
        }
    ];
    var nv1 = new Niivue({
        show3Dcrosshair: true,
        dragAndDropEnabled: true,
        onLocationChange: handleIntensityChange,
        // backColor: [51, 51, 51, 1],
    });
    nv1.opts.crosshairGap = 12
    nv1.opts.dragMode = nv1.dragModes.pan
    nv1.opts.yoke3Dto2DZoom = true
    alphaSlider.oninput = function () {
        nv1.volumes[1].cal_min = this.value / 255;
        nv1.updateGLVolume();
    }
    nv1.opts.isResizeCanvas = true;
    canvas.classList.add("hidden");

    nv1.attachTo("gl1");
    nv1.setSliceType(nv1.sliceTypeMultiplanar);
    nv1.loadVolumes(volumeList1);

    let dataOptsVol = document.createElement("div");
    dataOptsVol.style.display = "none";
    dataOptsVol.textContent = "neurovlm";
    dataOptsVol.style.color = "white";
    dataOptsVol.style.textShadow = "0px 2px 8px black, 0px 1px 8px black";
    dataOptsVol.style.fontSize = "24pt";
    dataOptsVol.style.fontWeight = "bold";
    dataOptsVol.id = "dataopts-vol";
    document.body.appendChild(dataOptsVol);

    const toggleOptions = document.querySelectorAll('.toggle-option');
    toggleOptions.forEach(option => {
        option.addEventListener('click', () => {

            toggleOptions.forEach(opt => opt.classList.remove('active'));
            option.classList.add('active');
            console.log(option.id);

            if (option.id == "mni"){

                let div1 = document.getElementsByName("w2figure0")[0].children[0];
                div1.classList.remove("visible");
                div1.classList.add("hidden");

                let div2 = document.getElementById("gl1");
                div2.classList.remove("hidden");
                div2.classList.add("visible");

                dataOptsVol.style.display = "flex";

                // Threshold slider
                let divSlider = document.getElementById("alphaSlider");
                divSlider.style.display = "flex";

                // Force reflow
                nv1.canvas.width = nv1.canvas.offsetWidth * nv1.uiData.dpr;
                nv1.canvas.height = nv1.canvas.offsetHeight * nv1.uiData.dpr;
                nv1.gl.viewport(0, 0, nv1.gl.canvas.width, nv1.gl.canvas.height);
                nv1.drawScene();


            } else{
                let div1 = document.getElementById("gl1");
                div1.classList.remove("visible");
                div1.classList.add("hidden");
                dataOptsVol.style.display = "none";

                let div2 = document.getElementsByName("w2figure0")[0].children[0];
                div2.classList.remove("hidden");
                div2.classList.add("visible");

                // Threshold slider
                let divSlider = document.getElementById("alphaSlider");
                divSlider.style.display = "none";

                // Force reflow
                void div2.offsetHeight;
            }
      });
    });

    async function waitForDivToHide(divId) {
        const div = document.getElementById(divId);
        return new Promise(resolve => {
            const observer = new MutationObserver(() => {
                if (div.style.display === "none") {
                    observer.disconnect(); // Stop observing
                    resolve();
                }
            });
            // Start observing changes to the 'style' attribute
            observer.observe(div, { attributes: true, attributeFilter: ["style"] });
            // If it's already hidden, resolve immediately
            if (div.style.display === "none") {
                observer.disconnect();
                resolve();
            }
        });
    }
    waitForDivToHide("ctmload").then(() => {
      setTimeout(() => {
        document.getElementById("loader").style.display = "none";
      }, 1000);
    });
    // // Inverse query
    // document.addEventListener("idxChanged", function (e) {
    //     const newIdx = e.detail;  // This is the updated float value
    //     run_inverse_query(newIdx);
    //     console.log(newIdx);
    // });
}
run();


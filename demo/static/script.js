const video = document.getElementById("video");

const canvas = document.getElementById("canvas");

const captureBtn = document.getElementById("captureBtn");

const imageInput = document.getElementById("imageInput");

const preview = document.getElementById("preview");

const loading = document.getElementById("loading");

const prediction = document.getElementById("prediction");

const score = document.getElementById("score");


async function startCamera(){

    try{

        const stream = await navigator.mediaDevices.getUserMedia({

            video:{
                width:1920,
                height:1080,
                facingMode:"environment"
            }

        });

        video.srcObject = stream;

    }
    catch(err){

        alert("Unable to access camera.");

        console.log(err);
    }
}

startCamera();


async function predictImage(file){

    loading.style.display="block";

    prediction.innerHTML="";

    score.innerHTML="";

    const formData=new FormData();

    formData.append("image",file);

    try{

        const response=await fetch("/predict",{

            method:"POST",

            body:formData

        });

        const data=await response.json();

        loading.style.display="none";

        if(!response.ok){

            prediction.innerHTML="ERROR";

            prediction.className="fake";

            score.innerHTML=data.error||"Prediction failed";

            return;
        }

        if(data.score>=0.5){

            prediction.innerHTML="PHOTO OF SCREEN";

            prediction.className="fake";
        }
        else{

            prediction.innerHTML="REAL PHOTO";

            prediction.className="real";
        }

        score.innerHTML="Score : "+Number(data.score).toFixed(3);

    }
    catch(err){

        loading.style.display="none";

        prediction.innerHTML="ERROR";

        prediction.className="fake";

        score.innerHTML="Could not reach server";

        console.log(err);
    }
}


captureBtn.addEventListener("click",()=>{

    canvas.width=video.videoWidth;

    canvas.height=video.videoHeight;

    const ctx=canvas.getContext("2d");

    ctx.drawImage(video,0,0);

    preview.src=canvas.toDataURL("image/jpeg");

    preview.style.display="block";

    video.style.display="none";

    canvas.toBlob(function(blob){

        predictImage(blob);

    },"image/jpeg",1.0);

});


imageInput.addEventListener("change",()=>{

    if(imageInput.files.length===0)
        return;

    const file=imageInput.files[0];

    preview.src=URL.createObjectURL(file);

    preview.style.display="block";

    video.style.display="none";

    predictImage(file);

});
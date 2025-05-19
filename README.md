
# Lightweight Sensor Fusion Network for Visual Perception

Robust perception with low computation is pivotal for safe navigation in modern robotics and autonomous driving. Yet most existing solutions rely on heavy post processing, two stage matching, which inflates latency, memory and engineering complexity. To address these limitations, We introduce an end-to-end fusion pipeline that exploits Lidar geometry and camera semantics without any post-hoc 2D/3D association.  
In this proposed pipeline, raw lidar sweeps are rasterized into a three-channel bird’s-eye-view(BEV) grid. The lightweight ResNet KFPN backbone then regresses fully metrised 7-DoF bounding boxes in a single, anchor free pass and 
within each box the closest inlier point provides an explicit forward range supplying minimum computation. The system emplyoes the compute aware preprocessing ROI cropping plus a one shot RANSAC ground filter cuts the point load by approx 60% while FP16 quantization shrinks the model from 73 MB to 51 MB. The Network benchmarked on KITTI dataset and the model attains 70.7, after optimization accuracy reduced minimal with 2.7 mAP drop with about 30%reduction in model size.  
Finally the complete pipeline is packaged as a ROS node, it streams depth aware overlays in real time, meeting the tight latency and memory budgets of embedded robotic and autonomous vehicle platforms.

## Demo

Insert gif or link to demo


## Installation

Install my-project with npm

```bash
  npm install my-project
  cd my-project
```
    
## Roadmap

- Additional browser support

- Add more integrations


## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```



## Usage/Examples

```javascript
import Component from 'my-project'

function App() {
  return <Component />
}
```


## Acknowledgements

 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)


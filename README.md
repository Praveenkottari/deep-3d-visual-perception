
# Lightweight Sensor Fusion Network for Visual Perception

Robust perception with low computation is pivotal for safe navigation in modern robotics and autonomous driving. Yet most existing solutions rely on heavy post processing, two stage matching, which inflates latency, memory and engineering complexity. To address these limitations, We introduce an end-to-end fusion pipeline that exploits Lidar geometry and camera semantics without any post-hoc 2D/3D association.  
In this proposed pipeline, raw lidar sweeps are rasterized into a three channel bird’s-eye-view(BEV) grid, then the lightweight ResNet KFPN backbone regresses fully metrised 7-DoF bounding boxes in a single anchor free pass and within each detection the closest inlier point provides an explicit forward range supplying minimum computation. This network employees the compute aware preprocessing, ROI cropping plus a one shot RANSAC ground filter that cuts the point load by \(\approx60\%\). While FP16 quantization shrinks the model by 30\%. The Network benchmarked on KITTI dataset and the model attains \(70.7\,\text{mAP}\) which is comparably higher than the baseline models. Finally, the complete pipeline is packaged as a ROS node, which streams depth aware overlays in real time, meeting the tight latency and memory budgets of embedded robotic and autonomous vehicle platforms.


![Flowchart](docs/Fusion-pipline.png)

<p align="center"><i>Figure : Dataset preparation work-flow</i></p>

## All below sections are coming soon.....
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


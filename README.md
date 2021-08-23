<!-- Instructions to run the program:

-> Store front and side images in the 'in' folder in the repo directory.
-> Change directory to hmr2/src
-> Run download_model.py if running for the first time.
-> Run final.py and enter height. Enter height as 0 to stop the program.
-> Output is stored in the 'out' folder in the repo directory.
-> accuracy.py in the repo directory can be used to check accuracy of the calculation if actual measurements 
    are stored in the actual_measure.json file in the 'in' folder. -->

# 2D Auto Measurement
Extraction of Human Body Measurements from 2D images for Clothing options

![](header.png)
![](sample_data\Pose_3d_model\side view.png)
![](sample_data\Pose_3d_model\front view.png)

## Instructions

Install requirements
```sh
pip install requirements.txt
```
Store front and side images in the 'in' folder in the repo directory.
Change directory to hmr2/src.
```sh
cd hmr2/src
```
Run download_model.py if running for the first time.
```sh
python download_model.py
```
Run final.py and enter height. Enter height as 0 to stop the program.
```sh
python final.py
```
Run accuracy.py to check accuracy of the calculation after storing actual measurements in the actual_measure.json file in the 'in' folder.
```sh
cd .. 
cd ..
python accuracy.py
```


## Usage example

A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.

_For more examples and usage, please refer to the [Wiki][wiki]._

## Development setup

Describe how to install all development dependencies and how to run an automated test-suite of some kind. Potentially do this for multiple platforms.

```sh
make install
npm test
```

## Release History

* 0.2.1
    * CHANGE: Update docs (module code remains unchanged)
* 0.2.0
    * CHANGE: Remove `setDefaultXYZ()`
    * ADD: Add `init()`
* 0.1.1
    * FIX: Crash when calling `baz()` (Thanks @GenerousContributorName!)
* 0.1.0
    * The first proper release
    * CHANGE: Rename `foo()` to `bar()`
* 0.0.1
    * Work in progress

## Meta

Your Name – [@YourTwitter](https://twitter.com/dbader_org) – YourEmail@example.com

Distributed under the XYZ license. See ``LICENSE`` for more information.

[https://github.com/yourname/github-link](https://github.com/dbader/)

## Contributing

1. Fork it (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

<!-- Markdown link & img dfn's -->
[npm-image]: https://img.shields.io/npm/v/datadog-metrics.svg?style=flat-square
[npm-url]: https://npmjs.org/package/datadog-metrics
[npm-downloads]: https://img.shields.io/npm/dm/datadog-metrics.svg?style=flat-square
[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
[wiki]: https://github.com/yourname/yourproject/wiki
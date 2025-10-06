# Land Cover Classification in Georgia with Machine Learning

### Testing Classification Algorithms in Google Earth Engine

Google Earth Engine is a platform for viewing, analyzing, and manipulating large spatial data without reading any files onto memory. For working with remotely sensed data, it is a gamechanger.

I wanted to explore running machine learning algorithms on remotely sensed data using the **ee** and **geemap** packages. I took a simple concept -- land cover classification -- and attempted to classify pixels around Augusta, Georgia using the United States National Land Cover Dataset for 2021.

The **ee.classifier** package handles supervised classification by traditional ML algorithms running in Earth Engine. These classifiers include CART, RandomForest, NaiveBayes and SVM. The general workflow for classification is:

1. Collect training data. Assemble features which have a property that stores the known class label and properties storing numeric values for the predictors.
2. Instantiate a classifier. Set its parameters if necessary.
3. Train the classifier using the training data.
4. Classify an image or feature collection.
5. Estimate classification error with independent validation data.

Source: https://developers.google.com/earth-engine/classification


I decided to classify using CART to start. Then I tried Random Forest, which appears to perform better.

### Import libraries


```python
# Raad in important libraries.
import ee
import geemap
import leaflet
import ipyleaflet
```


```python
# Connect to Google Earth Engine.
ee.Authenticate()
ee.Initialize(project='ee-matthewrungaro')
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



### Create an interactive map


```python
Map = geemap.Map()
Map
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    Map(center=[0, 0], controls=(WidgetControl(options=['position', 'transparent_bg'], position='topright', transp…



### Add data to the map

I selected a point near the city of Savannah, Georgia. I wanted to a diverse set of land cover classes. This area contains **farmland** (81 - Pasture/Hay, 82 - Cultivated Crops), **wetlands** (90 - Woody Wetlands, 95 - Emergency Herbaceious Wetlands), **forests** (42 - Evergreen Forest, 43 - Mixed Forest), and of course, **developed land** (22 - Developed, Low Intensity, 23 - Developed, Medium Intensity, etc.).

I selected a point, then captured a Landsat image in 2021 with the least amount of cloud cover.


```python
# Select a point
point = ee.Geometry.Point([-81.048316, 32.335148])

# Capture an image from Landsat from 2021 where the image with the lowest
# cloud cover is selected. This will help to train the model.
image = (
    ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    .filterBounds(point)
    .filterDate("2021-01-01", "2021-12-31")
    .sort("CLOUD_COVER")
    .first()
    .select("SR_B[1-7]")
)

# Visualize the image.
vis_params = {"min": 1, "max": 65455, "bands": ["SR_B5", "SR_B4", "SR_B3"]}

Map.centerObject(point, 8)
Map.addLayer(image, vis_params, "Landsat-8")
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



### Check image properties

We can see the date and cloud cover in the following functions. The cloud cover is very low. Should not negatively impact training.


```python
ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    '2021-12-13'




```python
# See percentage cloud cover.
image.get("CLOUD_COVER").getInfo()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    1.26



### Make training dataset

There are several ways to create a region for generating the training dataset. I'm going with the default -- using the image footprint. There are other ways:

- Draw a shape (e.g., rectangle) on the map and the use `region = Map.user_roi`
- Define a geometry, such as `region = ee.Geometry.Rectangle([-122.6003, 37.4831, -121.8036, 37.8288])`
- Create a buffer zone around a point, such as `region = ee.Geometry.Point([-122.4439, 37.7538]).buffer(10000)`


I'll select the 2021 release of the [National Land Cover Dataset](https://developers.google.com/earth-engine/datasets/catalog/USGS_NLCD_RELEASES_2021_REL_NLCD) for training. I'll clip it to the Landsat image I selected.


```python
nlcd = ee.ImageCollection("USGS/NLCD_RELEASES/2021_REL/NLCD").select("landcover").first()
nlcd = nlcd.clip(image.geometry())
Map.addLayer(nlcd, {}, "NLCD")
Map
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    Map(bottom=26842.0, center=[32.335148, -81.048316], controls=(WidgetControl(options=['position', 'transparent_…




```python
# Make the training dataset. Sample 5000 pixels as points.
points = nlcd.sample(
    **{
        "region": image.geometry(),
        "scale": 30,
        "numPixels": 5000,
        "seed": 0,
        "geometries": True,  # Set this to False to ignore geometries
    }
)

Map.addLayer(points, {}, "training", False)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
print(points.size().getInfo())
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    5000
    

We can see an example of one of our points here. We select the first point, and it has a landcover class of 42. This is **evergreen forest**.


```python
points.first()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






<div><style>:root {
  --font-color-primary: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --font-color-secondary: var(--jp-content-font-color2, rgba(0, 0, 0, 0.7));
  --font-color-accent: rgba(123, 31, 162, 1);
  --border-color: var(--jp-border-color2, #e0e0e0);
  --background-color: var(--jp-layout-color0, white);
  --background-color-row-even: var(--jp-layout-color1, white);
  --background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme="dark"],
body[data-theme="dark"],
body.vscode-dark {
  --font-color-primary: rgba(255, 255, 255, 1);
  --font-color-secondary: rgba(255, 255, 255, 0.7);
  --font-color-accent: rgb(173, 132, 190);
  --border-color: #2e2e2e;
  --background-color: #111111;
  --background-color-row-even: #111111;
  --background-color-row-odd: #313131;
}

.eerepr {
  padding: 1em;
  line-height: 1.5em;
  min-width: 300px;
  max-width: 1200px;
  overflow-y: scroll;
  max-height: 600px;
  border: 1px solid var(--border-color);
  font-family: monospace;
  font-size: 14px;
}

.eerepr li {
  list-style-type: none;
  margin: 0;
}

.eerepr ul {
  padding-left: 1.5em !important;
  margin: 0;
}

.eerepr > ul {
  padding-left: 0 !important;
}

.eerepr summary {
  color: var(--font-color-secondary);
  cursor: pointer;
  margin: 0;
}

.eerepr summary:hover {
  color: var(--font-color-primary);
  background-color: var(--background-color-row-odd)
}

.ee-k {
  color: var(--font-color-accent);
  margin-right: 6px;
}

.ee-v {
  color: var(--font-color-primary);
}

.eerepr details > summary::before {
  content: '▼';
  display: inline-block;
  margin-right: 6px;
  transition: transform 0.2s;
  transform: rotate(-90deg);
}

.eerepr details[open] > summary::before {
  transform: rotate(0deg);
}

.eerepr details summary::-webkit-details-marker {
  display:none;
}

.eerepr details summary {
  list-style-type: none;
}
</style><div class='eerepr'><ul><li><details><summary>Feature (Point, 1 property)</summary><ul><li><span class='ee-k'>type:</span><span class='ee-v'>Feature</span></li><li><span class='ee-k'>id:</span><span class='ee-v'>0</span></li><li><details><summary>geometry: Point (-81.82, 33.51)</summary><ul><li><span class='ee-k'>type:</span><span class='ee-v'>Point</span></li><li><details><summary>coordinates: [-81.8234200055685, 33.51152485343634]</summary><ul><li><span class='ee-k'>0:</span><span class='ee-v'>-81.8234200055685</span></li><li><span class='ee-k'>1:</span><span class='ee-v'>33.51152485343634</span></li></ul></details></li></ul></details></li><li><details><summary>properties: Object (1 property)</summary><ul><li><span class='ee-k'>landcover:</span><span class='ee-v'>42</span></li></ul></details></li></ul></details></li></ul></div></div>



### Train the classifier


```python
# Use these bands for prediction.
bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]


# This property of the table stores the land cover labels.
label = "landcover"

# Overlay the points on the imagery to get training.
sample = image.select(bands).sampleRegions(
    **{"collection": points, "properties": [label], "scale": 30}
)

# Adds a column of deterministic pseudorandom numbers.
sample = sample.randomColumn()

split = 0.7

training = sample.filter(ee.Filter.lt("random", split))
validation = sample.filter(ee.Filter.gte("random", split))

# Train a CART classifier with default parameters.
trained = ee.Classifier.smileCart().train(training, label, bands)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
print(training.first().getInfo())
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    {'type': 'Feature', 'geometry': None, 'id': '1_0', 'properties': {'SR_B1': 8591, 'SR_B2': 8992, 'SR_B3': 9630, 'SR_B4': 10290, 'SR_B5': 15220, 'SR_B6': 16694, 'SR_B7': 13432, 'landcover': 71, 'random': 0.36280327385355104}}
    

### Classify the image


```python
# Classify the image with the same bands used for training.
result = image.select(bands).classify(trained)

# # Display the clusters with random colors.
Map.addLayer(result.randomVisualizer(), {}, "classified")
Map
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    Map(bottom=6994.0, center=[31.240985378021307, -79.48608398437501], controls=(WidgetControl(options=['position…



The rivers seem to be classified correctly. Unclear if anything else is, however.

### Render categorical map




```python
class_values = nlcd.get("landcover_class_values").getInfo()
class_values
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]




```python
class_palette = nlcd.get("landcover_class_palette").getInfo()
class_palette
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    ['466b9f',
     'd1def8',
     'dec5c5',
     'd99282',
     'eb0000',
     'ab0000',
     'b3ac9f',
     '68ab5f',
     '1c5f2c',
     'b5c58f',
     'ccb879',
     'dfdfc2',
     'dcd939',
     'ab6c28',
     'b8d9eb',
     '6c9fb8']




```python
landcover = result.set("classification_class_values", class_values)
landcover = landcover.set("classification_class_palette", class_palette)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
Map.addLayer(landcover, {}, "Land Cover (CART)")
Map
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    Map(bottom=6994.0, center=[31.240985378021307, -79.48608398437501], controls=(WidgetControl(options=['position…



### Visualize the result


```python
print("Change layer opacity:")
cluster_layer = Map.layers[-1]
cluster_layer.interact(opacity=(0, 1, 0.1))
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    Change layer opacity:
    




    Box(children=(FloatSlider(value=1.0, description='opacity', max=1.0),))



### Add a legend to the map


```python
Map.add_legend(builtin_legend="NLCD")
Map
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    Map(bottom=6994.0, center=[31.240985378021307, -79.48608398437501], controls=(WidgetControl(options=['position…



The classifier seems to be identifying many more "90 - Woody Wetlands" than what the NLCD map has categorized.


```python
train_accuracy = trained.confusionMatrix()
train_accuracy.accuracy().getInfo()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    1




```python
validated = validation.classify(trained)

test_accuracy = validated.errorMatrix("landcover", "classification")
test_accuracy.accuracy().getInfo()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    0.34324324324324323



With only 34% classified accurately, perhaps we should try another model.

## Try Random Forest Classification

Random forest classification is an ensemble classification method that takes a decision tree implementation (like in CART) and attempt to "boost" it. "Each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample) from the training set. Furthermore, when splitting each node during the construction of a tree, the best split is found through an exhaustive search of the feature values of either all input features or a random subset."

[Ensembles - Random Forest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests). *Scikit-Learn, 2025*.


```python
# Use these bands for prediction.
bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]

# This property of the table stores the land cover labels.
label = "landcover"

# Overlay the points on the imagery to get training.
sample = image.select(bands).sampleRegions(
    **{"collection": points, "properties": [label], "scale": 30}
)

# Adds a column of deterministic pseudorandom numbers.
sample = sample.randomColumn()

split = 0.7

training = sample.filter(ee.Filter.lt("random", split))
validation = sample.filter(ee.Filter.gte("random", split))


# Train.
classifier  = ee.Classifier.smileRandomForest(10).train(training, label, bands)

# Classify the image with the same bands used for training.
result_rf = image.select(bands).classify(classifier )

# # Display the clusters with random colors.
Map.addLayer(result_rf.randomVisualizer(), {}, "classified_rf")
Map

landcover_rf = result_rf.set("classification_class_values", class_values)
landcover_rf = landcover_rf.set("classification_class_palette", class_palette)

Map.addLayer(landcover_rf, {}, "Land cover RF")
Map
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    Map(bottom=6994.0, center=[31.240985378021307, -79.48608398437501], controls=(WidgetControl(options=['position…



Random forest appears to do a better job than CART in the main urban area in this image. Divisions between land and water appear clearer. Urban features like major roads over wetlands are hard to discern with CART. They appear relatively intact under random forest.

## Accuracy Assessment


```python
train_accuracy = classifier.confusionMatrix()
train_accuracy.accuracy().getInfo()

```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    0.9365936878021041




```python
train_accuracy.kappa().getInfo()

```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    0.9235766908863354




```python
validated = validation.classify(classifier)
validated.first().getInfo()

```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    {'type': 'Feature',
     'geometry': None,
     'id': '0_0',
     'properties': {'SR_B1': 7604,
      'SR_B2': 7675,
      'SR_B3': 8342,
      'SR_B4': 8117,
      'SR_B5': 14192,
      'SR_B6': 10262,
      'SR_B7': 8634,
      'classification': 42,
      'landcover': 42,
      'random': 0.9162731201450195}}




```python
test_accuracy = validated.errorMatrix("landcover", "classification")
test_accuracy.accuracy().getInfo()

```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>






    0.43445945945945946



With random forest, the test accuracy goes from 34% to 43%. This is an improvement and validates what I was seeing--under CART, many of the pixels seem randomly distributed with no rhyme or reason. With random forest, structures inherit to actual land cover begin to appear.

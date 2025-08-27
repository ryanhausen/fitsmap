'use strict';

// Author: Mingyu Li (lmytime [at] hotmail.com)
// Date: 2024 Nov 11
// Description:
// ====================
// This script ensures that the x and y values of `translate3d` transformations are rounded to integers
// whenever the move or zoom action on the map ends.
// This helps to prevent the appearance of "white sub-pixel image borders" that occur due to floating point rendering
// inaccuracies, which is a common issue especially in Google Chrome.
// Note that this script can lead to slight shifts in the map position after a move or zoom operation, but it is generally acceptable.
// ====================

// Function to round the x and y components of the `translate3d` CSS transformation to integer values.
// This function is executed after the map ends a movement or zoom operation.
const integerTranslateMapPane = function (event) {
    // Obtain the map pane element, which contains the current translation transformation information.
    var mapPane = event.target.getPane('mapPane');
    var transformStyle = mapPane.style.transform;

    // Use a regular expression to extract the x, y, and z values from the `translate3d` transformation.
    var xyzMatches = transformStyle.match(/translate3d\((-?\d+(\.\d+)?)px, (-?\d+(\.\d+)?)px, (-?\d+(\.\d+)?)px\)/);

    // If the `transform` style includes valid `translate3d` values, proceed with rounding the x and y.
    if (xyzMatches) {
        // Convert the matched x, y, and z values to floating point numbers, then round them to the nearest integer.
        var xTranslateInt = Math.round(parseFloat(xyzMatches[1])); // Round the x component to the nearest integer
        var yTranslateInt = Math.round(parseFloat(xyzMatches[3])); // Round the y component to the nearest integer
        var zTranslateInt = Math.round(parseFloat(xyzMatches[5])); // Round the z component to the nearest integer (typically 0)

        // Update the `transform` style of the map pane to use the rounded x, y, and z values.
        mapPane.style.transform = `translate3d(${xTranslateInt}px, ${yTranslateInt}px, ${zTranslateInt}px)`;
    }
};

// Register event listeners on the map to execute the integer rounding function whenever the map ends a movement (`moveend`)
// or zoom operation (`zoomend`). This ensures that any slight inaccuracies from floating point values are corrected immediately.
// map.on("moveend", integerTranslateMapPane);
// map.on("zoomend", integerTranslateMapPane);
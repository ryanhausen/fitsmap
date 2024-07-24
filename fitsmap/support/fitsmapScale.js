// https://stackoverflow.com/a/62093918
L.Control.FitsMapScale = L.Control.Scale.extend({
    options: {
        unitsArePixels: true,
        pixelScale: 1.0,
    },

    _updateMetric: function (maxPixels) {
        const pixels = this._getRoundNum(maxPixels);

        let distance = this.options.unitsArePixels ? pixels : pixels * this.options.pixelScale;
        distance = parseFloat(distance.toFixed(2));

        let units;
        if (this.options.unitsArePixels){
            units = 'px';
        } else{
            if (distance > 60) {
                distance = distance / 60;
                units = "'";
            }
            else {
                units = '"';
            }
        }

        const label = distance + units;
        this._updateScale(this._mScale, label, pixels / maxPixels);
    }
});

L.control.fitsmapScale = function (options) {
    const scaleOpts = {imperial:false, updateWhenIdle:true};
    return new L.Control.FitsMapScale(L.extend(scaleOpts, options));
}


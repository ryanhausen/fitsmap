// https://stackoverflow.com/a/62093918
L.Control.FitsMapScale = L.Control.Scale.extend({
    options: {
        isPixels: true,
        pixelScale: 1.0,
    },

    _updateMetric: function (maxPixels) {
        const pixels = this._getRoundNum(maxPixels);

        let distance = this.options.isPixels ? pixels : pixels * pixelScale;

        let units;
        if (this.options.isPixels){
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


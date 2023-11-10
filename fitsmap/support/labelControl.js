// =============================================================================
// Control that displays the label text
L.Control.Label = L.Control.extend({
    options : {
        displayHTML : '',
        isRADec: true,
        nDecimals: 5,
        title: '',
        latlng: {lat: 0, lng: 0}
    },

    makeText: function (coord1, coord2) {
        const unit1 = this.options.isRADec ? "RA" : "x";
        const unit2 = this.options.isRADec ? "Dec" : "y";
        let labelText = '<div class="label-control">';
        labelText += this.options.title!="" ? `<p class="label-control">${this.options.title}</p>` : "";
        labelText += `<p class="label-control">${unit1}: ${coord1.toFixed(this.options.nDecimals)} ${unit2}: ${coord2.toFixed(this.options.nDecimals)}</p>`
        labelText += '</div>';
        return labelText;
    },


    onAdd: function () {
        const div = L.DomUtil.create('div', 'label-control');
        div.innerHTML = this.makeText(0.0, 0.0);
        this.options.displayHTML = div;

        return div;
    },

    update: function (latlng) {
        if (this.options.isRADec) {
            raDec = pixToSky(latlng);
            this.options.displayHTML.innerHTML = this.makeText(raDec[0], raDec[1]);
        } else {
            this.options.displayHTML.innerHTML = this.makeText(latlng.lat, latlng.lng);
        }

    },

    onRemove: function (map) {
    },
});

L.control.label = function (opts) {
    return new L.Control.Label(opts);
}
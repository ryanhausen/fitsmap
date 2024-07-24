'use strict';

L.GridLayer.TiledMarkers = L.GridLayer.extend({

    options: {
        tileURL: "",
        strokeColor: "#4C72B0",
        fillColor: "#4C72B0",
        fillOpacity: 0.2,
        strokeOpacity: 1.0,
        scaleRadius: 1.0,
        radius: 10,
        nCols: 1,
        catalogColumns: [],
    },

    initialize: function(options) {
        L.setOptions(this, options);
        this.tilePointCache = {}
    },

    convertJSONtoHTMLTable: function(json) {
        // these are extra keys used for the search function
        // const dontDisplay = ["fm_y", "fm_x", "fm_cat"]
        // const nItems = Object.keys(json).length - dontDisplay.length;
        const nItems = this.options.catalogColumns.length;
        const rowsPerCol = this.options.rowsPerColumn;
        const values = json["v"];

        // let nCols = Math.floor(nItems / (Number.isFinite(rowsPerCol) ? rowsPerCol : nItems));
        let nCols = this.options.nCols <= 0 ? 1 : this.options.nCols;
        // if (Number.isFinite(rowsPerCol) && nItems % rowsPerCol > 0) {
        //     nCols += 1;
        // }

        let html = "<span>Catalog Information</span>" +
                    "<table class='catalog-table'>";

        let colCounter = 0;
        html += "<tr>";
        for (let i = 0; i < nItems; i++){
            let key = this.options.catalogColumns[i];
            let value = values[i];

            html += `<td><b>${key}:<b></td><td>${value}</td>`;

            colCounter += 1;
            if (colCounter == nCols){
                colCounter = 0;
                html += "</tr>";
            }
        }

        html += "</table>";

        return html;
    },

    renderPopupContents: function(_this, marker) {
        const popup = marker.getPopup();

        fetch(marker.options.assetPath)
        .then((response) => {
            if (!response.ok){
                console.log(response);
                throw new Error("Failed to fetch JSON", response);
            }
            return response.arrayBuffer();
        }).then(buffer =>{
            return cbor.decodeAll(buffer);
        }).then(json => {
            popup.setContent(_this.convertJSONtoHTMLTable(json[0])).update();
        })
        .catch((error) => {
            console.log("ERROR in Popup Rendering", error);
        });

        // https://stackoverflow.com/a/51749619/2691018
        document.querySelector(".leaflet-popup-pane").addEventListener("load", function(event) {
            const tagName = event.target.tagName,
            popup = map._popup;
            // Also check if flag is already set.
            if (tagName === "IMG" && popup && !popup._updated) {
                popup._updated = true; // Set flag to prevent looping.
                popup.update();
            }
        }, true);
        return "Loading...";
    },

    createClusterIcon: function(src) {
        const latlng = L.latLng(src.global_y, src.global_x);
        if (!src.cluster){

            const p = L.popup({ maxWidth: "auto" })
                     .setLatLng(latlng)
                     .setContent((layer) => this.renderPopupContents(this, layer));

            if (src.a==-1){
                return L.circleMarker(latlng, {
                    color: this.options.strokeColor,
                    fillColor: this.options.fillColor,
                    assetPath: `catalog_assets/${src.cat_path}/${src.catalog_id}.cbor`,
                    opacity: this.options.strokeOpacity,
                    fillOpacity: this.options.fillOpacity,
                    radius: this.options.radius * this.options.scaleRadius,
                }).bindPopup(p);
            } else {
                return L.ellipse(latlng, [src.a * this.options.scaleRadius, src.b * this.options.scaleRadius], src.theta, {
                    color: this.options.strokeColor,
                    fillColor: this.options.fillColor,
                    opacity: this.options.strokeOpacity,
                    fillOpacity: this.options.fillOpacity,
                    assetPath: `catalog_assets/${src.cat_path}/${src.catalog_id}.cbor`
                }).bindPopup(p);
            }
        }

        // Create an icon for a cluster
        const count = src.point_count;
        const size =
            count < 100 ? "small" :
            count < 1000 ? "medium" :
            count < 1000000 ? "large" : "x-large";
        const icon = L.divIcon({
            html: `<div><span>${src.point_count_abbreviated}</span></div>`,
            className: `marker-cluster marker-cluster-${size}`,
            iconSize: L.point(40, 40)
        });
        return L.marker(latlng, {icon}).bindPopup(`${src.global_y}, ${src.global_x}`);
    },


    parseTileResponse: function(key, response) {
        if (response.status==200){
            response.arrayBuffer().then(data => {
                if (!this.tilePointCache[key]){
                    this.tilePointCache[key] = [];
                }
                const pbuf = new Pbf(data);
                const vTileData = new VectorTile(pbuf);
                const points = vTileData.layers.Points;
                for (let i = 0; i < points.length; i++){
                    let point = points.feature(i);
                    this.tilePointCache[key].push(
                        this.createClusterIcon(point.properties).addTo(map)
                    );
                }
            }).catch(err => console.log(err));
        }
    },


    createTile: function (coords) {
        const offset_y = 2**coords.z - coords.y - 1
        const offset_x = 2**coords.z - coords.x - 1
        const resourceURL = this.options.tileURL
                          .replace("{z}", `${coords.z}`)
                          .replace("{y}", `${offset_y}`)
                          .replace("{x}", `${coords.x}`)

        const key = `${coords.z},${coords.y},${coords.x}`
        fetch(resourceURL).then((r) => this.parseTileResponse(key, r)).catch((error) => {
            console.log(error);
        });

        return L.DomUtil.create('canvas', 'leaflet-tile');
    },

    clearItems: function(e){
        const key = `${e.coords.z},${e.coords.y},${e.coords.x}`
        if (this.tilePointCache[key]){
            while (this.tilePointCache[key].length){
                let p = this.tilePointCache[key].pop().remove();
                p = null;
            }
        }
    }
});

L.gridLayer.tiledMarkers = function(opts) {
    const layer =  new L.GridLayer.TiledMarkers(opts);
    layer.on("tileunload", layer.clearItems);
    return layer;
};

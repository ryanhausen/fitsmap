'use strict';

L.GridLayer.TiledMarkers = L.GridLayer.extend({
    tilePointCache: {},

    options: {
        tileURL: "",
        color: "#4C72B0",
    },

    initialize: function(options) {
        L.setOptions(this, options);
    },

    createClusterIcon: function(src) {
        const latlng = L.latLng(src.global_y, src.global_x);
        if (!src.cluster){
            const width = (((src.widest_col * 10) * src.n_cols) + 10).toString() + "em";
            const include_img = src.include_img ? 2 : 1;
            const height = ((src.n_rows + 1) * 15 * (include_img)).toString() + "em";

            const p = L.popup({ maxWidth: "auto" })
                     .setLatLng(latlng)
                     .setContent("<iframe src='catalog_assets/" + src.cat_path + "/" + src.catalog_id + ".html' width='" + width + "' height='" + height + "'></iframe>");

            if (src.a==-1){
                return L.circleMarker(latlng, {
                    color: this.options.color
                }).bindPopup(p);
            } else {
                return L.ellipse(latlng, [src.a, src.b], (src.theta * (180/Math.PI) * -1), {
                    color: this.options.color
                }).bindPopup(p);
            }

        }

        // Create an icon for a cluster
        const count = src.point_count;
        const size =
            count < 100 ? "small" :
            count < 1000 ? "medium" : "large";
        const icon = L.divIcon({
            html: `<div><span>${  src.point_count_abbreviated  }</span></div>`,
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


    createTile: function (coords, done) {
        //const bnds = this._pxBoundsToTileRange(this._getTiledPixelBounds(map.getCenter()));
        //const max_y = bnds.max.y;
        //const offset_y = max_y - coords.y;
        const offset_y = 2**coords.z - coords.y - 1
        const offset_x = 2**coords.z - coords.x - 1
        const resourceURL = this.options.tileURL
                          .replace("{z}", `${coords.z}`)
                          .replace("{y}", `${offset_y}`)
                          .replace("{x}", `${coords.x + 1}`)

        const key = `${coords.z},${coords.y},${coords.x}`
        fetch(resourceURL).then((r) => this.parseTileResponse(key, r)).catch((error) => {
            console.log(error);
        });

        return L.DomUtil.create('canvas', 'leaflet-tile');
    },

    clearItemsOLD: function(e){
        const key = `${e.coords.z},${e.coords.y},${e.coords.x}`
        // If we change zooms delete all from previous zoom
        if (e.coords.z != map.getZoom() && map.getZoom() <= this.options.maxNativeZoom){
            const keys = Object.keys(this.tilePointCache).filter(k => k[0]==`${e.coords.z}`);
            for (let i = 0; i < keys.length; i++){
                let key = keys[i];
                for (let j = 0; j < this.tilePointCache[key].length; j++){
                    this.tilePointCache[key][j].remove();
                }
                delete this.tilePointCache[key];
            }
            
        // If we're panning, then only delete what we have too
        } else {
            
            const key = `${e.coords.z},${e.coords.y},${e.coords.x}`
            if (key in this.tilePointCache){
                for (let i = 0; i < this.tilePointCache[key].length; i++){
                    this.tilePointCache[key][i].remove();
                }
                delete this.tilePointCache[key];
            }
        }
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

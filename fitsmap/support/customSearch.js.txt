'use strict';

function buildCustomSearch(catalogPaths, zoom){

    function searchByFnames(text, callbackResponse) {

        const searchCandidates = catalogPaths.map(c => c + text + ".json");

        // console.log("candidates", searchCandidates);
        Promise.all(
            searchCandidates.map(f => {
                return fetch(f).then(response => {
                    if (response.ok){
                        //console.log(response);
                        return response.json();
                    } else {
                        //console.log(response);
                        return Promise.resolve(null);
                    }
                })
            })
        ).then(values => {
            // console.log("values", values);
            const searchResults = values.filter(v => v).map(v => {
                // console.log(v);
                return {title:v.id + ":" + v.fm_cat, loc:[v.fm_y,v.fm_x]}
            });
            // console.log(searchResults);
            callbackResponse(searchResults);
        });

        return {	//called to stop previous requests on map move
            abort: function() {
                console.log('aborted request:'+ text);
            }
        };
    }

    function searchFound(latlng, title, map){
        map.setView(latlng, zoom);
    }

    const searchControl = new L.Control.Search({
        sourceData: searchByFnames,
        textPlaceholder: "Enter Catalog ID",
        textErr: "Catalog ID not found",
        markerLocation: true,
        hideMarkerOnCollapse: true,
        moveToLocation: searchFound
    });

    return searchControl;
}
